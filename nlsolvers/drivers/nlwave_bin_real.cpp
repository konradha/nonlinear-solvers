#include "../common/include/unified/cmdline_parser.hpp"
#include "../common/include/unified/solver_factory.hpp"
#include "../common/include/unified/solver_base.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    try {
        // Parse command line arguments
        nlsolvers::SolverParams params = nlsolvers::CommandLineParser::parse(argc, argv);
        
        // Check if the system type is real-space
        if (params.system_type == nlsolvers::SolverParams::SystemType::NLSE_CUBIC ||
            params.system_type == nlsolvers::SolverParams::SystemType::NLSE_CUBIC_QUINTIC ||
            params.system_type == nlsolvers::SolverParams::SystemType::NLSE_SATURATING) {
            throw std::runtime_error("This binary only supports real-space equations");
        }
        
        // Create appropriate solver
        auto solver = nlsolvers::SolverFactory::create_realspace_solver<double>(params);
        
        // Initialize solver
        solver->initialize(params.grid_sizes, params.domain_sizes);
        
        // Load initial condition
        solver->load_initial_condition(params.input_file);
        
        // Load anisotropy if provided
        if (params.anisotropy_file) {
            solver->load_anisotropy(*params.anisotropy_file);
        }
        
        // Load focusing/coupling parameter if provided
        if (params.focusing_file) {
            auto realspace_solver = dynamic_cast<nlsolvers::RealSpaceSolverBase<double>*>(solver.get());
            if (realspace_solver) {
                realspace_solver->load_coupling(*params.focusing_file);
            }
        }
        
        // Load initial velocity if provided
        if (params.initial_velocity_file) {
            auto realspace_solver = dynamic_cast<nlsolvers::RealSpaceSolverBase<double>*>(solver.get());
            if (realspace_solver) {
                realspace_solver->load_initial_velocity(*params.initial_velocity_file);
            } else {
                throw std::runtime_error("Initial velocity file provided but solver does not support it");
            }
        } else {
            std::cerr << "Warning: No initial velocity file provided. Using default (zeros)." << std::endl;
        }
        
        // Set velocity output file if provided
        if (params.velocity_output_file) {
            auto realspace_solver = dynamic_cast<nlsolvers::RealSpaceSolverBase<double>*>(solver.get());
            if (realspace_solver) {
                realspace_solver->set_velocity_output_file(*params.velocity_output_file);
            }
        }
        
        // Start timing
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Run simulation
        solver->run(params.T, params.nt, params.num_snapshots, params.output_file);
        
        // End timing
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "Simulation completed in " << duration.count() / 1000.0 << " seconds." << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
