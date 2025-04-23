#ifndef REAL_WAVE_SOLVER_HPP
#define REAL_WAVE_SOLVER_HPP

#include "base_solver.hpp"
#include "execution_context.hpp"
#include "host_execution_context.hpp"
#include "../common/include/util.hpp"
#include "../common/include/laplacians.hpp"
#include "../host/include/boundaries.hpp"
#include "../host/include/boundaries_3d.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <string>
#include <vector>
#include <optional>
#include <stdexcept>

/**
 * @brief Base class for real-valued wave solvers.
 * 
 * This abstract class implements common functionality for real-valued wave solvers.
 * It is templated on the dimension (2D or 3D).
 * 
 * @tparam Dim The dimension of the problem (2 or 3)
 */
template <int Dim>
class RealWaveSolver : public BaseSolver<Dim, double> {
public:
    using Parameters = typename BaseSolver<Dim, double>::Parameters;
    
    /**
     * @brief Constructor
     * 
     * @param params The solver parameters
     * @param device_type The device type ("host" or "cuda")
     */
    RealWaveSolver(const Parameters& params, const std::string& device_type = "host")
        : BaseSolver<Dim, double>(params), device_type_(device_type) {
        
        // Create execution context
        try {
            execution_context_ = ExecutionContext<double>::create(device_type);
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to create execution context: " + std::string(e.what()));
        }
    }
    
    /**
     * @brief Destructor
     */
    ~RealWaveSolver() override = default;
    
    /**
     * @brief Initialize the solver
     * 
     * @param initial_u_file Path to the file with initial condition
     * @param initial_v_file Optional path to the file with initial velocity
     * @return true if initialization was successful, false otherwise
     */
    bool initialize(const std::string& initial_u_file, 
                   const std::optional<std::string>& initial_v_file = std::nullopt) override {
        try {
            // Load initial condition
            std::vector<uint32_t> input_shape;
            this->u_ = read_from_npy<double>(initial_u_file, input_shape);
            
            // Check dimensions
            if constexpr (Dim == 2) {
                if (input_shape.size() != 2 || 
                    input_shape[0] != this->params_.n || 
                    input_shape[1] != this->params_.n) {
                    throw std::runtime_error("Invalid initial condition dimensions");
                }
            } else if constexpr (Dim == 3) {
                if (input_shape.size() != 3 || 
                    input_shape[0] != this->params_.n || 
                    input_shape[1] != this->params_.n || 
                    input_shape[2] != this->params_.n) {
                    throw std::runtime_error("Invalid initial condition dimensions");
                }
            }
            
            // Load initial velocity if provided, otherwise use zero
            if (initial_v_file) {
                v_ = read_from_npy<double>(initial_v_file.value(), input_shape);
                
                // Check dimensions
                if constexpr (Dim == 2) {
                    if (input_shape.size() != 2 || 
                        input_shape[0] != this->params_.n || 
                        input_shape[1] != this->params_.n) {
                        throw std::runtime_error("Invalid initial velocity dimensions");
                    }
                } else if constexpr (Dim == 3) {
                    if (input_shape.size() != 3 || 
                        input_shape[0] != this->params_.n || 
                        input_shape[1] != this->params_.n || 
                        input_shape[2] != this->params_.n) {
                        throw std::runtime_error("Invalid initial velocity dimensions");
                    }
                }
            } else {
                v_ = Eigen::VectorXd::Zero(this->grid_size_);
            }
            
            // Initialize past state (u_past = u - dt * v)
            u_past_ = this->u_ - this->dt_ * v_;
            
            // Load coefficients
            if (!this->load_coefficients()) {
                throw std::runtime_error("Failed to load coefficients");
            }
            
            // Build Laplacian
            if (!this->build_laplacian()) {
                throw std::runtime_error("Failed to build Laplacian");
            }
            
            // Initialize execution context
            if (!execution_context_->initialize(this->L_)) {
                throw std::runtime_error("Failed to initialize execution context");
            }
            
            // Initialize snapshots
            u_snapshots_.resize(this->params_.snapshots * this->grid_size_);
            v_snapshots_.resize(this->params_.snapshots * this->grid_size_);
            
            // Store initial snapshot
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> u_snapshots_mat(
                u_snapshots_.data(), this->params_.snapshots, this->grid_size_);
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> v_snapshots_mat(
                v_snapshots_.data(), this->params_.snapshots, this->grid_size_);
            
            u_snapshots_mat.row(0) = this->u_.transpose();
            v_snapshots_mat.row(0) = v_.transpose();
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Initialization failed: " << e.what() << std::endl;
            return false;
        }
    }
    
    /**
     * @brief Run the full simulation
     * 
     * @param trajectory_file Path to save the trajectory
     * @param velocity_file Optional path to save the velocity
     * @return true if the simulation was successful, false otherwise
     */
    bool run(const std::string& trajectory_file,
            const std::optional<std::string>& velocity_file = std::nullopt) override {
        try {
            for (int i = 1; i < this->params_.nt; ++i) {
                if (!step()) {
                    throw std::runtime_error("Step failed at iteration " + std::to_string(i));
                }
                
                if (i % this->freq_ == 0) {
                    int snapshot_idx = i / this->freq_;
                    if (snapshot_idx < this->params_.snapshots) {
                        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> u_snapshots_mat(
                            u_snapshots_.data(), this->params_.snapshots, this->grid_size_);
                        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> v_snapshots_mat(
                            v_snapshots_.data(), this->params_.snapshots, this->grid_size_);
                        
                        u_snapshots_mat.row(snapshot_idx) = this->u_.transpose();
                        v_snapshots_mat.row(snapshot_idx) = v_.transpose();
                    }
                }
            }
            
            // Save trajectory
            std::vector<uint32_t> shape;
            if constexpr (Dim == 2) {
                shape = {static_cast<uint32_t>(this->params_.snapshots), 
                         static_cast<uint32_t>(this->params_.n), 
                         static_cast<uint32_t>(this->params_.n)};
            } else if constexpr (Dim == 3) {
                shape = {static_cast<uint32_t>(this->params_.snapshots), 
                         static_cast<uint32_t>(this->params_.n), 
                         static_cast<uint32_t>(this->params_.n), 
                         static_cast<uint32_t>(this->params_.n)};
            }
            
            save_to_npy(trajectory_file, u_snapshots_, shape);
            
            // Save velocity if requested
            if (velocity_file) {
                save_to_npy(velocity_file.value(), v_snapshots_, shape);
            }
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Run failed: " << e.what() << std::endl;
            return false;
        }
    }
    
    /**
     * @brief Get the current state
     * 
     * @return const reference to the current state vector
     */
    const Eigen::VectorXd& get_state() const override {
        return this->u_;
    }
    
    /**
     * @brief Get the current velocity
     * 
     * @return const reference to the current velocity vector
     */
    const Eigen::VectorXd& get_velocity() const override {
        return v_;
    }
    
protected:
    std::string device_type_;                      // Device type ("host" or "cuda")
    std::unique_ptr<ExecutionContext<double>> execution_context_; // Execution context
    
    Eigen::VectorXd v_;                            // Current velocity
    Eigen::VectorXd u_past_;                       // Past state
    Eigen::VectorXd buf_;                          // Buffer for computations
    
    Eigen::VectorXd u_snapshots_;                  // Snapshots of the state
    Eigen::VectorXd v_snapshots_;                  // Snapshots of the velocity
    
    /**
     * @brief Load anisotropy and focussing coefficients
     * 
     * @return true if loading was successful, false otherwise
     */
    bool load_coefficients() override {
        try {
            // Load anisotropy coefficient
            if (this->params_.anisotropy_file) {
                std::vector<uint32_t> shape;
                this->c_ = read_from_npy<double>(this->params_.anisotropy_file.value(), shape);
                
                // Check dimensions
                if constexpr (Dim == 2) {
                    if (shape.size() != 2 || 
                        shape[0] != this->params_.n || 
                        shape[1] != this->params_.n) {
                        std::cerr << "Invalid anisotropy coefficient dimensions, using default" << std::endl;
                        this->c_ = Eigen::VectorXd::Ones(this->grid_size_);
                    }
                } else if constexpr (Dim == 3) {
                    if (shape.size() != 3 || 
                        shape[0] != this->params_.n || 
                        shape[1] != this->params_.n || 
                        shape[2] != this->params_.n) {
                        std::cerr << "Invalid anisotropy coefficient dimensions, using default" << std::endl;
                        this->c_ = Eigen::VectorXd::Ones(this->grid_size_);
                    }
                }
            } else {
                this->c_ = Eigen::VectorXd::Ones(this->grid_size_);
            }
            
            // Load focussing coefficient
            if (this->params_.focussing_file) {
                std::vector<uint32_t> shape;
                this->m_ = read_from_npy<double>(this->params_.focussing_file.value(), shape);
                
                // Check dimensions
                if constexpr (Dim == 2) {
                    if (shape.size() != 2 || 
                        shape[0] != this->params_.n || 
                        shape[1] != this->params_.n) {
                        std::cerr << "Invalid focussing coefficient dimensions, using default" << std::endl;
                        this->m_ = Eigen::VectorXd::Ones(this->grid_size_);
                    }
                } else if constexpr (Dim == 3) {
                    if (shape.size() != 3 || 
                        shape[0] != this->params_.n || 
                        shape[1] != this->params_.n || 
                        shape[2] != this->params_.n) {
                        std::cerr << "Invalid focussing coefficient dimensions, using default" << std::endl;
                        this->m_ = Eigen::VectorXd::Ones(this->grid_size_);
                    }
                }
            } else {
                this->m_ = Eigen::VectorXd::Ones(this->grid_size_);
            }
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Failed to load coefficients: " << e.what() << std::endl;
            return false;
        }
    }
    
    /**
     * @brief Build the Laplacian matrix
     * 
     * @return true if building was successful, false otherwise
     */
    bool build_laplacian() override {
        try {
            if constexpr (Dim == 2) {
                if (this->params_.anisotropy_file) {
                    this->L_ = (-(build_anisotropic_laplacian_noflux<double>(
                        this->params_.n - 2, this->params_.n - 2, this->dx_, this->dx_, this->c_))).eval();
                } else {
                    this->L_ = (-(build_laplacian_noflux<double>(
                        this->params_.n - 2, this->params_.n - 2, this->dx_, this->dx_))).eval();
                }
            } else if constexpr (Dim == 3) {
                if (this->params_.anisotropy_file) {
                    this->L_ = (-(build_anisotropic_laplacian_noflux_3d<double>(
                        this->params_.n - 2, this->params_.n - 2, this->params_.n - 2, 
                        this->dx_, this->dx_, this->dx_, this->c_))).eval();
                } else {
                    this->L_ = (-(build_laplacian_noflux_3d<double>(
                        this->params_.n - 2, this->params_.n - 2, this->params_.n - 2, 
                        this->dx_, this->dx_, this->dx_))).eval();
                }
            }
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Failed to build Laplacian: " << e.what() << std::endl;
            return false;
        }
    }
    
    /**
     * @brief Apply boundary conditions
     * 
     * @return true if applying was successful, false otherwise
     */
    bool apply_boundary_conditions() override {
        try {
            if constexpr (Dim == 2) {
                neumann_bc_no_velocity<double>(this->u_, this->params_.n, this->params_.n);
            } else if constexpr (Dim == 3) {
                neumann_bc_no_velocity_3d<double>(this->u_, this->params_.n, this->params_.n, this->params_.n);
            }
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Failed to apply boundary conditions: " << e.what() << std::endl;
            return false;
        }
    }
    
    /**
     * @brief Update velocity from current and past state
     * 
     * @return true if updating was successful, false otherwise
     */
    bool update_velocity() {
        try {
            v_ = (this->u_ - u_past_) / this->dt_;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Failed to update velocity: " << e.what() << std::endl;
            return false;
        }
    }
    
    /**
     * @brief Compute the nonlinear term
     * 
     * This method should be implemented by derived classes to compute
     * the nonlinear term for the specific equation.
     * 
     * @param u Current state
     * @param result Output vector for the nonlinear term
     * @return true if computation was successful, false otherwise
     */
    virtual bool compute_nonlinear_term(const Eigen::VectorXd& u, Eigen::VectorXd& result) = 0;
};

#endif // REAL_WAVE_SOLVER_HPP
