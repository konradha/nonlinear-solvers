#ifndef SOLVER_FACTORY_HPP
#define SOLVER_FACTORY_HPP

#include "base_solver.hpp"
#include "klein_gordon_solver.hpp"
#include "sine_gordon_solver.hpp"
#include "nlse_cubic_solver.hpp"
#include "../common/include/cmdline_parser.hpp"

#include <memory>
#include <string>
#include <stdexcept>

/**
 * @brief Factory class for creating solvers.
 * 
 * This class provides factory methods for creating solvers based on command-line arguments.
 */
class SolverFactory {
public:
    /**
     * @brief Create a real-valued wave solver
     * 
     * @param args Command-line arguments
     * @return std::unique_ptr<BaseSolver<Dim, double>> A unique pointer to the solver
     */
    template <int Dim>
    static std::unique_ptr<BaseSolver<Dim, double>> create_real_wave_solver(const CommandLineArgs& args) {
        // Convert device type enum to string
        std::string device_type = (args.device == DeviceType::HOST) ? "host" : "cuda";
        
        // Create solver parameters
        typename BaseSolver<Dim, double>::Parameters params(
            args.n, args.L, args.T, args.nt, args.snapshots);
        params.anisotropy_file = args.anisotropy_file;
        params.focussing_file = args.focussing_file;
        
        // Create solver based on system type
        switch (args.system_type) {
            case SystemType::KLEIN_GORDON:
                return std::make_unique<KleinGordonSolver<Dim>>(params, device_type);
            case SystemType::SINE_GORDON:
                return std::make_unique<SineGordonSolver<Dim>>(params, device_type);
            default:
                throw std::runtime_error("Unsupported system type for real-valued wave solver");
        }
    }
    
    /**
     * @brief Create a complex-valued wave solver
     * 
     * @param args Command-line arguments
     * @return std::unique_ptr<BaseSolver<Dim, std::complex<double>>> A unique pointer to the solver
     */
    template <int Dim>
    static std::unique_ptr<BaseSolver<Dim, std::complex<double>>> create_complex_wave_solver(const CommandLineArgs& args) {
        // Convert device type enum to string
        std::string device_type = (args.device == DeviceType::HOST) ? "host" : "cuda";
        
        // Create solver parameters
        typename BaseSolver<Dim, std::complex<double>>::Parameters params(
            args.n, args.L, args.T, args.nt, args.snapshots);
        params.anisotropy_file = args.anisotropy_file;
        params.focussing_file = args.focussing_file;
        
        // Create solver based on system type
        switch (args.system_type) {
            case SystemType::NLSE_CUBIC:
                return std::make_unique<NLSECubicSolver<Dim>>(params, device_type);
            default:
                throw std::runtime_error("Unsupported system type for complex-valued wave solver");
        }
    }
};

#endif // SOLVER_FACTORY_HPP
