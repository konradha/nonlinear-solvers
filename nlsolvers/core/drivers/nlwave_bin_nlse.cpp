#include "../include/base_solver.hpp"
#include "../include/solver_factory.hpp"
#include "../../common/include/cmdline_parser.hpp"

#include <iostream>
#include <memory>
#include <string>
#include <optional>
#include <complex>

int main(int argc, char** argv) {
    try {
        // Parse command-line arguments
        CommandLineArgs args = CommandLineParser::parse_nlse_args(argc, argv);
        
        // Create solver based on dimension
        if (args.dim == 2) {
            // Create 2D solver
            auto solver = SolverFactory::create_complex_wave_solver<2>(args);
            
            // Initialize solver
            if (!solver->initialize(args.initial_u_file)) {
                std::cerr << "Failed to initialize solver" << std::endl;
                return 1;
            }
            
            // Run simulation
            if (!solver->run(args.trajectory_file)) {
                std::cerr << "Failed to run simulation" << std::endl;
                return 1;
            }
        } else if (args.dim == 3) {
            // Create 3D solver
            auto solver = SolverFactory::create_complex_wave_solver<3>(args);
            
            // Initialize solver
            if (!solver->initialize(args.initial_u_file)) {
                std::cerr << "Failed to initialize solver" << std::endl;
                return 1;
            }
            
            // Run simulation
            if (!solver->run(args.trajectory_file)) {
                std::cerr << "Failed to run simulation" << std::endl;
                return 1;
            }
        } else {
            std::cerr << "Unsupported dimension: " << args.dim << std::endl;
            return 1;
        }
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
