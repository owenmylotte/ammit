#include <petsc.h>
#include <environment/download.hpp>
#include <environment/runEnvironment.hpp>
#include <fstream>
#include <io/hdf5MultiFileSerializer.hpp>
#include <localPath.hpp>
#include <mathFunctions/functionFactory.hpp>
#include <memory>
#include <parameters/petscPrefixOptions.hpp>
#include <utilities/mpiUtilities.hpp>
#include <yamlParser.hpp>
#include "builder.hpp"
#include "domain/boxMesh.hpp"
#include "domain/modifiers/distributeWithGhostCells.hpp"
#include "domain/modifiers/ghostBoundaryCells.hpp"
#include "eos/perfectGas.hpp"
#include "finiteVolume/boundaryConditions/ghost.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "finiteVolume/compressibleFlowSolver.hpp"
#include "finiteVolume/fluxCalculator/ausm.hpp"
#include "io/interval/fixedInterval.hpp"
#include "monitors/curveMonitor.hpp"
#include "monitors/timeStepMonitor.hpp"
#include "parameters/mapParameters.hpp"
#include "utilities/petscUtilities.hpp"

typedef struct {
    PetscReal gamma;
    PetscReal length;
    PetscReal rhoL;
    PetscReal uL;
    PetscReal pL;
    PetscReal rhoR;
    PetscReal uR;
    PetscReal pR;
} InitialConditions;

const char* replacementInputPrefix = "-yaml::";

int main(int argc, char** argv) {
    // initialize petsc and mpi
    ablate::environment::RunEnvironment::Initialize(&argc, &argv);
    ablate::utilities::PetscUtilities::Initialize();

    {
        // check to see if we should print options
        char filename[PETSC_MAX_PATH_LEN] = "";
        PetscBool fileSpecified = PETSC_FALSE;
        PetscOptionsGetString(nullptr, nullptr, "--input", filename, PETSC_MAX_PATH_LEN, &fileSpecified) >> ablate::utilities::PetscUtilities::checkError;
        if (!fileSpecified) {
            throw std::invalid_argument("the --input must be specified");
        }

        // locate or download the file
        std::filesystem::path filePath;
        if (ablate::environment::Download::IsUrl(filename)) {
            ablate::environment::Download downloader(filename);
            filePath = downloader.Locate();
        } else {
            cppParser::LocalPath locator(filename);
            filePath = locator.Locate();
        }

        if (!std::filesystem::exists(filePath)) {
            throw std::invalid_argument("unable to locate input file: " + filePath.string());
        }
        {
            // build options from the command line
            auto yamlOptions = std::make_shared<ablate::parameters::PetscPrefixOptions>(replacementInputPrefix);

            // create the yaml parser
            std::shared_ptr<cppParser::Factory> parser = std::make_shared<cppParser::YamlParser>(filePath, yamlOptions->GetMap());

            // setup the monitor
            auto setupEnvironmentParameters = parser->GetByName<ablate::parameters::Parameters>("environment");
            ablate::environment::RunEnvironment::Setup(*setupEnvironmentParameters, filePath);

            // Copy over the input file
            int rank;
            MPI_Comm_rank(PETSC_COMM_WORLD, &rank) >> ablate::utilities::MpiUtilities::checkError;
            if (rank == 0) {
                std::filesystem::path inputCopy = ablate::environment::RunEnvironment::Get().GetOutputDirectory() / filePath.filename();
                std::ofstream stream(inputCopy);
                stream.close();
            }

            // run with the parser
            // get the global arguments
            auto globalArguments = parser->GetByName<ablate::parameters::Parameters>("arguments");
            if (globalArguments) {
                globalArguments->Fill(nullptr);
            }

            // create a time stepper
            auto timeStepper = parser->Get(cppParser::ArgumentIdentifier<ablate::solver::TimeStepper>{.inputName = "timestepper"});

            // Check to see if a single or multiple solvers were specified
            if (parser->Contains("solver")) {
                auto solver = parser->GetByName<ablate::solver::Solver>("solver");
                auto solverMonitors = parser->GetFactory("solver")->GetByName<std::vector<ablate::monitors::Monitor>>("monitors", std::vector<std::shared_ptr<ablate::monitors::Monitor>>());
                timeStepper->Register(solver, solverMonitors);
            }

            // Add in other solvers
            auto solverList = parser->GetByName<std::vector<ablate::solver::Solver>>("solvers", std::vector<std::shared_ptr<ablate::solver::Solver>>());
            std::vector<std::shared_ptr<ablate::monitors::Monitor>> monitorList;
            if (!solverList.empty()) {
                auto solverFactorySequence = parser->GetFactorySequence("solvers");

                // initialize the flow for each
                for (std::size_t i = 0; i < solverFactorySequence.size(); i++) {
                    auto& solver = solverList[i];
                    auto solverMonitors = solverFactorySequence[i]->GetByName<std::vector<ablate::monitors::Monitor>>("monitors", std::vector<std::shared_ptr<ablate::monitors::Monitor>>());
                    timeStepper->Register(solver, solverMonitors);
                    monitorList = solverMonitors;
                }
            }

            timeStepper->Initialize();  //! Registers all of the monitors and initializes them within the newly created domain.

            /**
             * Loop through the time steps that are saved in the input file directory.
             * Set up the domain with the value of each time step and call the monitor save functions.
             */
            for (int i = 0; i < 1; i++) {
                /**
                 * Make a serializer that is dedicated to saving only.
                 */
                // Register the solver with the serializer
                ablate::io::Hdf5MultiFileSerializer postSerializer(i, nullptr);  // Create the same type of serializer that is in the input file
                for (auto& monitor : monitorList) {
                    auto serializable = std::dynamic_pointer_cast<ablate::io::Serializable>(monitor);
                    if (serializable && serializable->Serialize()) {
                        postSerializer.Register(serializable);  //! Register all of the monitors with the serializer so that they can be written out through it.
                    }
                }

                // TODO: Use the restart code to get the time step that is wanted now.

                for (auto& monitor : monitorList) {
                    /**
                     * Loop through each monitor and call the serializer on it.
                     */
                    monitor->CallMonitor(timeStepper->GetTS(), i, 0, nullptr); //! This saves the information to the HDF5 file.
                }
            }

            // check for unused parameters
            auto unusedValues = parser->GetUnusedValues();
            if (!unusedValues.empty()) {
                std::cout << "WARNING: The following input parameters were not used:" << std::endl;
                for (auto unusedValue : unusedValues) {
                    std::cout << unusedValue << std::endl;
                }
            }
        }
    }

    ablate::environment::RunEnvironment::Finalize();
    return 0;
}