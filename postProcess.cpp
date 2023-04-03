#include <petsc.h>
#include <environment/runEnvironment.hpp>
#include <fstream>
#include <io/hdf5MultiFileSerializer.hpp>
#include <localPath.hpp>
#include <memory>
#include <parameters/petscPrefixOptions.hpp>
#include <utilities/mpiUtilities.hpp>
#include <yamlParser.hpp>
#include "builder.hpp"
#include "finiteVolume/compressibleFlowSolver.hpp"
#include "monitors/curveMonitor.hpp"
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
        char directoryName[PETSC_MAX_PATH_LEN] = "";
        PetscBool fileSpecified = PETSC_FALSE;
        PetscOptionsGetString(nullptr, nullptr, "--directory", directoryName, PETSC_MAX_PATH_LEN, &fileSpecified) >> ablate::utilities::PetscUtilities::checkError;
        if (!fileSpecified) {
            throw std::invalid_argument("the --directory must be specified");  //! Take a directory instead of an input file. Use whatever yaml file in the directory.
        }
        // Get the path of the simulation to post process.
        std::filesystem::path directoryPath;
        cppParser::LocalPath locator(directoryName);
        directoryPath = locator.Locate();

        // Iterate through the files in the directory and look for a yaml file to use as the input file.
        std::filesystem::path inputFilePath;
        std::filesystem::path restartFilePath;
        for (auto const& dir_entry : std::filesystem::directory_iterator{directoryPath}) {
            if (dir_entry.path().extension().string() == ".yaml") inputFilePath = dir_entry.path();   // If it's a yaml file, then make this the input file path.
            if (dir_entry.path().extension().string() == ".rst") restartFilePath = dir_entry.path();  // If it's a restart file, then make this the restart file path.
        }

        if (!std::filesystem::exists(inputFilePath)) {
            throw std::invalid_argument("unable to locate input file: " + inputFilePath.string());
        }

        {
            // build options from the command line
            auto yamlOptions = std::make_shared<ablate::parameters::PetscPrefixOptions>(replacementInputPrefix);

            // create the yaml parser
            std::shared_ptr<cppParser::Factory> parser = std::make_shared<cppParser::YamlParser>(inputFilePath, yamlOptions->GetMap());

            // setup the monitor
            //            auto setupEnvironmentParameters = parser->GetByName<ablate::parameters::Parameters>("environment");
            //            ablate::environment::RunEnvironment::Setup(*setupEnvironmentParameters, inputFilePath);

            // Get the restart file from the working directory so that the max sequence number is known
            auto yaml = YAML::LoadFile(restartFilePath);                         //! Use whatever restart file in the input directory.
            PetscInt maxSequenceNumber = yaml["sequenceNumber"].as<PetscInt>();  //! Store the sequence number of the input file

            // Copy over the input file
            int rank;
            MPI_Comm_rank(PETSC_COMM_WORLD, &rank) >> ablate::utilities::MpiUtilities::checkError;
            if (rank == 0) {
                std::filesystem::path inputCopy = ablate::environment::RunEnvironment::Get().GetOutputDirectory() / inputFilePath.filename();
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

            // Register the solver with the serializer
            auto postSerializer = ablate::io::Hdf5MultiFileSerializer(0, nullptr);
            for (auto& monitor : monitorList) {
                auto serializable = std::dynamic_pointer_cast<ablate::io::Serializable>(monitor);
                if (serializable && serializable->Serialize()) {
                    postSerializer.Register(serializable);  //! Register all of the monitors with the serializer so that they can be written out through it.
                }
            }

            /**
             * Loop through the time steps that are saved in the input file directory.
             * Set up the domain with the value of each time step and call the monitor save functions.
             */
            for (int i = 0; i < maxSequenceNumber; i++) {
                for (auto& monitor : monitorList) {
                    auto serializable = std::dynamic_pointer_cast<ablate::io::Serializable>(monitor);
                    postSerializer.RestoreFromSequence(i, serializable);       //! Use the restart code to get the time step that is wanted now.
                                                                                //! Loop through each monitor and call the serializer on it.
                    monitor->CallMonitor(timeStepper->GetTS(), i, 0, timeStepper->GetSolutionVector());  //! This saves the information to the HDF5 file.
                    postSerializer.GetSerializeFunction();
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