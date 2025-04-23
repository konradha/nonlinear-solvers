#ifndef KLEIN_GORDON_SOLVER_HPP
#define KLEIN_GORDON_SOLVER_HPP

#include "real_wave_solver.hpp"
#include <Eigen/Dense>
#include <string>
#include <optional>

/**
 * @brief Klein-Gordon equation solver.
 * 
 * This class implements the Klein-Gordon equation:
 * u_tt - div(c(x)grad(u)) + m(x) * u = 0
 * 
 * @tparam Dim The dimension of the problem (2 or 3)
 */
template <int Dim>
class KleinGordonSolver : public RealWaveSolver<Dim> {
public:
    using Parameters = typename RealWaveSolver<Dim>::Parameters;
    
    /**
     * @brief Constructor
     * 
     * @param params The solver parameters
     * @param device_type The device type ("host" or "cuda")
     */
    KleinGordonSolver(const Parameters& params, const std::string& device_type = "host")
        : RealWaveSolver<Dim>(params, device_type) {
        
        // Initialize buffer
        this->buf_.resize(this->grid_size_);
    }
    
    /**
     * @brief Destructor
     */
    ~KleinGordonSolver() override = default;
    
    /**
     * @brief Perform a single time step using the Gautschi method
     * 
     * @return true if the step was successful, false otherwise
     */
    bool step() override {
        try {
            // Compute nonlinear term
            if (!this->compute_nonlinear_term(this->u_, this->buf_)) {
                return false;
            }
            
            // Apply Gautschi method
            Eigen::VectorXd buf2;
            if (!this->execution_context_->sqrt_multiply(this->u_, buf2, this->dt_)) {
                return false;
            }
            
            buf2 = -this->m_.cwiseProduct(buf2);
            
            if (!this->execution_context_->sinc2_sqrt_half(buf2, buf2, this->dt_)) {
                return false;
            }
            
            Eigen::VectorXd u_cpy = this->u_;
            
            if (!this->execution_context_->cos_sqrt_multiply(this->u_, this->u_, this->dt_)) {
                return false;
            }
            
            this->u_ = 2.0 * this->u_ - this->u_past_ + this->dt_ * this->dt_ * buf2;
            this->u_past_ = u_cpy;
            
            // Apply boundary conditions
            if (!this->apply_boundary_conditions()) {
                return false;
            }
            
            // Update velocity
            if (!this->update_velocity()) {
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
     * @brief Compute the nonlinear term for the Klein-Gordon equation
     * 
     * For the Klein-Gordon equation, the nonlinear term is simply u.
     * 
     * @param u Current state
     * @param result Output vector for the nonlinear term
     * @return true if computation was successful, false otherwise
     */
    bool compute_nonlinear_term(const Eigen::VectorXd& u, Eigen::VectorXd& result) override {
        try {
            result = u;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Failed to compute nonlinear term: " << e.what() << std::endl;
            return false;
        }
    }
};

#endif // KLEIN_GORDON_SOLVER_HPP
