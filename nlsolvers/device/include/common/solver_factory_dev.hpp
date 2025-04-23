#ifndef SOLVER_FACTORY_DEV_HPP
#define SOLVER_FACTORY_DEV_HPP

#include "cmdline_parser_dev.hpp"
#include "../nlse/nlse_solver_base_dev.hpp"
#include "../realspace/realspace_solver_base_dev.hpp"

// Forward declarations for specific solvers
namespace device {
    template <typename Scalar_t, int Dim> class NLSECubicSolver;
    template <typename Scalar_t, int Dim> class NLSECubicQuinticSolver;
    template <typename Scalar_t, int Dim> class NLSESaturatingSolver;
    template <typename Scalar_t, int Dim> class KGSolver;
    template <typename Scalar_t, int Dim> class SGSingleSolver;
    template <typename Scalar_t, int Dim> class SGDoubleSolver;
    template <typename Scalar_t, int Dim> class SGHyperbolicSolver;
    template <typename Scalar_t, int Dim> class Phi4Solver;
}

namespace device {

class SolverFactory {
public:
    template <typename Scalar_t>
    static std::unique_ptr<NLSESolverBase<Scalar_t, 2>> create_nlse_solver_2d(const SolverParams& params) {
        std::unique_ptr<NLSESolverBase<Scalar_t, 2>> solver;
        
        switch (params.equation_type) {
            case SolverParams::EquationType::NLSE_CUBIC:
                solver = std::make_unique<NLSECubicSolver<Scalar_t, 2>>();
                break;
            case SolverParams::EquationType::NLSE_CUBIC_QUINTIC:
                solver = std::make_unique<NLSECubicQuinticSolver<Scalar_t, 2>>(params.sigma1, params.sigma2);
                break;
            case SolverParams::EquationType::NLSE_SATURATING:
                solver = std::make_unique<NLSESaturatingSolver<Scalar_t, 2>>();
                break;
            default:
                throw std::runtime_error("Invalid NLSE equation type for 2D");
        }
        
        // Set method
        switch (params.method) {
            case SolverParams::Method::SS2:
                solver->set_method(NLSESolverBase<Scalar_t, 2>::Method::SS2);
                break;
            case SolverParams::Method::SEWI:
                solver->set_method(NLSESolverBase<Scalar_t, 2>::Method::SEWI);
                break;
            default:
                throw std::runtime_error("Invalid method for NLSE solver");
        }
        
        return solver;
    }
    
    template <typename Scalar_t>
    static std::unique_ptr<NLSESolverBase<Scalar_t, 3>> create_nlse_solver_3d(const SolverParams& params) {
        std::unique_ptr<NLSESolverBase<Scalar_t, 3>> solver;
        
        switch (params.equation_type) {
            case SolverParams::EquationType::NLSE_CUBIC:
                solver = std::make_unique<NLSECubicSolver<Scalar_t, 3>>();
                break;
            case SolverParams::EquationType::NLSE_CUBIC_QUINTIC:
                solver = std::make_unique<NLSECubicQuinticSolver<Scalar_t, 3>>(params.sigma1, params.sigma2);
                break;
            case SolverParams::EquationType::NLSE_SATURATING:
                solver = std::make_unique<NLSESaturatingSolver<Scalar_t, 3>>();
                break;
            default:
                throw std::runtime_error("Invalid NLSE equation type for 3D");
        }
        
        // Set method
        switch (params.method) {
            case SolverParams::Method::SS2:
                solver->set_method(NLSESolverBase<Scalar_t, 3>::Method::SS2);
                break;
            case SolverParams::Method::SEWI:
                solver->set_method(NLSESolverBase<Scalar_t, 3>::Method::SEWI);
                break;
            default:
                throw std::runtime_error("Invalid method for NLSE solver");
        }
        
        return solver;
    }
    
    template <typename Scalar_t>
    static std::unique_ptr<RealSpaceSolverBase<Scalar_t, 2>> create_realspace_solver_2d(const SolverParams& params) {
        std::unique_ptr<RealSpaceSolverBase<Scalar_t, 2>> solver;
        
        switch (params.equation_type) {
            case SolverParams::EquationType::KG:
                solver = std::make_unique<KGSolver<Scalar_t, 2>>();
                break;
            case SolverParams::EquationType::SG_SINGLE:
                solver = std::make_unique<SGSingleSolver<Scalar_t, 2>>();
                break;
            case SolverParams::EquationType::SG_DOUBLE:
                solver = std::make_unique<SGDoubleSolver<Scalar_t, 2>>();
                break;
            case SolverParams::EquationType::SG_HYPERBOLIC:
                solver = std::make_unique<SGHyperbolicSolver<Scalar_t, 2>>();
                break;
            case SolverParams::EquationType::PHI4:
                solver = std::make_unique<Phi4Solver<Scalar_t, 2>>();
                break;
            default:
                throw std::runtime_error("Invalid real-space equation type for 2D");
        }
        
        // Set method
        switch (params.method) {
            case SolverParams::Method::GAUTSCHI:
                solver->set_method(RealSpaceSolverBase<Scalar_t, 2>::Method::GAUTSCHI);
                break;
            case SolverParams::Method::STORMER_VERLET:
                solver->set_method(RealSpaceSolverBase<Scalar_t, 2>::Method::STORMER_VERLET);
                break;
            default:
                throw std::runtime_error("Invalid method for real-space solver");
        }
        
        return solver;
    }
    
    template <typename Scalar_t>
    static std::unique_ptr<RealSpaceSolverBase<Scalar_t, 3>> create_realspace_solver_3d(const SolverParams& params) {
        std::unique_ptr<RealSpaceSolverBase<Scalar_t, 3>> solver;
        
        switch (params.equation_type) {
            case SolverParams::EquationType::KG:
                solver = std::make_unique<KGSolver<Scalar_t, 3>>();
                break;
            case SolverParams::EquationType::SG_SINGLE:
                solver = std::make_unique<SGSingleSolver<Scalar_t, 3>>();
                break;
            case SolverParams::EquationType::SG_DOUBLE:
                solver = std::make_unique<SGDoubleSolver<Scalar_t, 3>>();
                break;
            case SolverParams::EquationType::SG_HYPERBOLIC:
                solver = std::make_unique<SGHyperbolicSolver<Scalar_t, 3>>();
                break;
            case SolverParams::EquationType::PHI4:
                solver = std::make_unique<Phi4Solver<Scalar_t, 3>>();
                break;
            default:
                throw std::runtime_error("Invalid real-space equation type for 3D");
        }
        
        // Set method
        switch (params.method) {
            case SolverParams::Method::GAUTSCHI:
                solver->set_method(RealSpaceSolverBase<Scalar_t, 3>::Method::GAUTSCHI);
                break;
            case SolverParams::Method::STORMER_VERLET:
                solver->set_method(RealSpaceSolverBase<Scalar_t, 3>::Method::STORMER_VERLET);
                break;
            default:
                throw std::runtime_error("Invalid method for real-space solver");
        }
        
        return solver;
    }
};

} // namespace device

#endif // SOLVER_FACTORY_DEV_HPP
