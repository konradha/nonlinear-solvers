#ifndef NLSE_CUBIC_SOLVER_HPP
#define NLSE_CUBIC_SOLVER_HPP

#include "complex_wave_solver.hpp"
#include <Eigen/Dense>
#include <string>
#include <optional>
#include <complex>

/**
 * @brief Cubic NLSE solver.
 * 
 * This class implements the cubic nonlinear Schrödinger equation:
 * i u_t + div(c(x)grad(u)) + m(x) * |u|^2 * u = 0
 * 
 * @tparam Dim The dimension of the problem (2 or 3)
 */
template <int Dim>
class NLSECubicSolver : public ComplexWaveSolver<Dim> {
public:
    using Parameters = typename ComplexWaveSolver<Dim>::Parameters;
    
    /**
     * @brief Constructor
     * 
     * @param params The solver parameters
     * @param device_type The device type ("host" or "cuda")
     */
    NLSECubicSolver(const Parameters& params, const std::string& device_type = "host")
        : ComplexWaveSolver<Dim>(params, device_type) {
    }
    
    /**
     * @brief Destructor
     */
    ~NLSECubicSolver() override = default;
    
    /**
     * @brief Perform a single time step using the Strang splitting method
     * 
     * @return true if the step was successful, false otherwise
     */
    bool step() override {
        try {
            // Compute density
            Eigen::VectorXd density = (this->u_.array().abs2()).matrix();
            
            // First half-step of nonlinear part
            this->rho_buf_ = (this->m_.cwiseProduct(density))
                .unaryExpr([this](double x) { 
                    return std::exp(std::complex<double>(0, 0.5 * this->dt_ * x)); 
                })
                .cwiseProduct(this->u_);
            
            // Full step of linear part
            if (!this->execution_context_->expm_multiply(this->rho_buf_, this->buf_, 
                                                       std::complex<double>(0, this->dt_))) {
                return false;
            }
            
            // Second half-step of nonlinear part
            density = (this->buf_.array().abs2()).matrix();
            
            this->u_ = (this->m_.cwiseProduct(density))
                .unaryExpr([this](double x) { 
                    return std::exp(std::complex<double>(0, 0.5 * this->dt_ * x)); 
                })
                .cwiseProduct(this->buf_);
            
            // Apply boundary conditions
            if (!this->apply_boundary_conditions()) {
                return false;
            }
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Step failed: " << e.what() << std::endl;
            return false;
        }
    }
    
protected:
    /**
     * @brief Compute the nonlinear term for the cubic NLSE
     * 
     * For the cubic NLSE, the nonlinear term is |u|^2 * u.
     * 
     * @param u Current state
     * @param result Output vector for the nonlinear term
     * @return true if computation was successful, false otherwise
     */
    bool compute_nonlinear_term(const Eigen::VectorXcd& u, Eigen::VectorXcd& result) override {
        try {
            Eigen::VectorXd density = (u.array().abs2()).matrix();
            result = this->m_.cwiseProduct(density).cwiseProduct(u);
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Failed to compute nonlinear term: " << e.what() << std::endl;
            return false;
        }
    }
};

#endif // NLSE_CUBIC_SOLVER_HPP
