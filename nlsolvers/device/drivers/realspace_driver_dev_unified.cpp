#include "../include/common/cmdline_parser_dev.hpp"
#include "../include/common/solver_factory_dev.hpp"
#include "../include/realspace/realspace_solver_base_dev.hpp"
#include "laplacians.hpp"
#include "util.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    try {
        // Parse command line arguments
        device::SolverParams params = device::CommandLineParser::parse_realspace(argc, argv);
        
        // Create appropriate solver based on dimension
        if (params.dimension == 2) {
            auto solver = device::SolverFactory::create_realspace_solver_2d<double>(params);
            
            // Initialize solver
            solver->initialize(params.grid_sizes, params.domain_sizes);
            
            // Load initial condition
            solver->load_initial_condition(params.input_file);
            
            // Load initial velocity
            if (params.velocity_file) {
                solver->load_initial_velocity(*params.velocity_file);
            } else {
                throw std::runtime_error("Initial velocity file is required for real-space equations");
            }
            
            // Load coupling parameter if provided
            if (params.m_file) {
                solver->load_coupling_parameter(*params.m_file);
            }
            
            // Set velocity output file if provided
            if (params.velocity_output_file) {
                solver->set_velocity_output_file(*params.velocity_output_file);
            }
            
            // Run simulation
            solver->run(params.T, params.nt, params.num_snapshots, params.output_file);
        } else if (params.dimension == 3) {
            auto solver = device::SolverFactory::create_realspace_solver_3d<double>(params);
            
            // Initialize solver
            solver->initialize(params.grid_sizes, params.domain_sizes);
            
            // Load initial condition
            solver->load_initial_condition(params.input_file);
            
            // Load initial velocity
            if (params.velocity_file) {
                solver->load_initial_velocity(*params.velocity_file);
            } else {
                throw std::runtime_error("Initial velocity file is required for real-space equations");
            }
            
            // Load coupling parameter if provided
            if (params.m_file) {
                solver->load_coupling_parameter(*params.m_file);
            }
            
            // Set velocity output file if provided
            if (params.velocity_output_file) {
                solver->set_velocity_output_file(*params.velocity_output_file);
            }
            
            // Run simulation
            solver->run(params.T, params.nt, params.num_snapshots, params.output_file);
        } else {
            throw std::runtime_error("Unsupported dimension");
        }
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
