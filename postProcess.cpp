#include <petsc.h>
#include <environment/runEnvironment.hpp>
#include <mathFunctions/functionFactory.hpp>
#include <memory>
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

static PetscErrorCode SetInitialCondition(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx) {
    InitialConditions *initialConditions = (InitialConditions *)ctx;

    if (x[0] < initialConditions->length / 2.0) {
        u[ablate::finiteVolume::CompressibleFlowFields::RHO] = initialConditions->rhoL;
        u[ablate::finiteVolume::CompressibleFlowFields::RHOU + 0] = initialConditions->rhoL * initialConditions->uL;

        PetscReal e = initialConditions->pL / ((initialConditions->gamma - 1.0) * initialConditions->rhoL);
        PetscReal et = e + 0.5 * PetscSqr(initialConditions->uL);
        u[ablate::finiteVolume::CompressibleFlowFields::RHOE] = et * initialConditions->rhoL;

    } else {
        u[ablate::finiteVolume::CompressibleFlowFields::RHO] = initialConditions->rhoR;
        u[ablate::finiteVolume::CompressibleFlowFields::RHOU + 0] = initialConditions->rhoR * initialConditions->uR;

        PetscReal e = initialConditions->pR / ((initialConditions->gamma - 1.0) * initialConditions->rhoR);
        PetscReal et = e + 0.5 * PetscSqr(initialConditions->uR);
        u[ablate::finiteVolume::CompressibleFlowFields::RHOE] = et * initialConditions->rhoR;
    }

    return 0;
}

static PetscErrorCode PhysicsBoundary_Euler(PetscReal time, const PetscReal *c, const PetscReal *n, const PetscScalar *a_xI, PetscScalar *a_xG, void *ctx) {
    InitialConditions *initialConditions = (InitialConditions *)ctx;

    if (c[0] < initialConditions->length / 2.0) {
        a_xG[ablate::finiteVolume::CompressibleFlowFields::RHO] = initialConditions->rhoL;

        a_xG[ablate::finiteVolume::CompressibleFlowFields::RHOU + 0] = initialConditions->rhoL * initialConditions->uL;

        PetscReal e = initialConditions->pL / ((initialConditions->gamma - 1.0) * initialConditions->rhoL);
        PetscReal et = e + 0.5 * PetscSqr(initialConditions->uL);
        a_xG[ablate::finiteVolume::CompressibleFlowFields::RHOE] = et * initialConditions->rhoL;
    } else {
        a_xG[ablate::finiteVolume::CompressibleFlowFields::RHO] = initialConditions->rhoR;

        a_xG[ablate::finiteVolume::CompressibleFlowFields::RHOU + 0] = initialConditions->rhoR * initialConditions->uR;

        PetscReal e = initialConditions->pR / ((initialConditions->gamma - 1.0) * initialConditions->rhoR);
        PetscReal et = e + 0.5 * PetscSqr(initialConditions->uR);
        a_xG[ablate::finiteVolume::CompressibleFlowFields::RHOE] = et * initialConditions->rhoR;
    }
    return 0;
    PetscFunctionReturn(0);
}

int main(int argc, char **argv) {
    // initialize petsc and mpi
    ablate::environment::RunEnvironment::Initialize(&argc, &argv);
    ablate::utilities::PetscUtilities::Initialize();

    {
        // define some initial conditions
        InitialConditions initialConditions{.gamma = 1.4, .length = 1.0, .rhoL = 1.0, .uL = 0.0, .pL = 1.0, .rhoR = 0.125, .uR = 0.0, .pR = .1};

        // setup the run environment
        ablate::parameters::MapParameters runEnvironmentParameters(std::map<std::string, std::string>{{"title", "clientExample"}});
        ablate::environment::RunEnvironment::Setup(runEnvironmentParameters);

        auto eos = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}}));

        // determine required fields for finite volume compressible flow
        std::vector<std::shared_ptr<ablate::domain::FieldDescriptor>> fieldDescriptors = {std::make_shared<ablate::finiteVolume::CompressibleFlowFields>(eos)};

        auto domain =
            std::make_shared<ablate::domain::BoxMesh>("simpleMesh",
                                                      fieldDescriptors,
                                                      std::vector<std::shared_ptr<ablate::domain::modifiers::Modifier>>{std::make_shared<ablate::domain::modifiers::DistributeWithGhostCells>(),
                                                                                                                        std::make_shared<ablate::domain::modifiers::GhostBoundaryCells>()},
                                                      std::vector<int>{100},
                                                      std::vector<double>{0.0},
                                                      std::vector<double>{initialConditions.length},
                                                      std::vector<std::string>{"NONE"} /*boundary*/,
                                                      false /*simplex*/,
                                                      ablate::parameters::MapParameters::Create({{"dm_refine", "2"}, {"dm_distribute", ""}}));

        // Set up the flow data
        auto parameters = std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"cfl", ".4"}});

        // Set the initial conditions for euler
        auto initialCondition = std::make_shared<ablate::mathFunctions::FieldFunction>("euler", ablate::mathFunctions::Create(SetInitialCondition, (void *)&initialConditions));

        // create a time stepper
        auto timeStepper = ablate::solver::TimeStepper(
            domain, ablate::parameters::MapParameters::Create({{"ts_adapt_type", "physicsConstrained"}, {"ts_max_steps", "600"}, {"ts_dt", "0.00000625"}}), {}, {initialCondition});

        auto boundaryConditions = std::vector<std::shared_ptr<ablate::finiteVolume::boundaryConditions::BoundaryCondition>>{
            std::make_shared<ablate::finiteVolume::boundaryConditions::Ghost>("euler", "wall left", 1, PhysicsBoundary_Euler, (void *)&initialConditions),
            std::make_shared<ablate::finiteVolume::boundaryConditions::Ghost>("euler", "wall right", 2, PhysicsBoundary_Euler, (void *)&initialConditions)};

        // Create a shockTube solver
        auto shockTubeSolver = std::make_shared<ablate::finiteVolume::CompressibleFlowSolver>("compressibleShockTube",
                                                                                              ablate::domain::Region::ENTIREDOMAIN,
                                                                                              nullptr /*options*/,
                                                                                              eos,
                                                                                              parameters,
                                                                                              nullptr /*transportModel*/,
                                                                                              std::make_shared<ablate::finiteVolume::fluxCalculator::Ausm>(),
                                                                                              boundaryConditions /*boundary conditions*/);

        // register the flowSolver with the timeStepper
        timeStepper.Register(
            shockTubeSolver,
            {std::make_shared<ablate::monitors::TimeStepMonitor>(), std::make_shared<ablate::monitors::CurveMonitor>(std::make_shared<ablate::io::interval::FixedInterval>(10), "outputCurve")});

        // Solve the time stepper
        timeStepper.Solve();
    }

    ablate::environment::RunEnvironment::Finalize();
    return 0;
}