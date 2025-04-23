#ifndef BASE_SOLVER_HPP
#define BASE_SOLVER_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <string>
#include <memory>
#include <vector>
#include <optional>

/**
 * @brief Base class for all nonlinear wave solvers.
 * 
 * This abstract class defines the common interface for all solvers.
 * It is templated on the dimension (2D or 3D) and the scalar type
 * (double for real waves, std::complex<double> for complex waves).
 * 
 * @tparam Dim The dimension of the problem (2 or 3)
 * @tparam Scalar_t The scalar type (double or std::complex<double>)
 */
template <int Dim, typename Scalar_t>
class BaseSolver {
public:
    /**
     * @brief Struct to hold common solver parameters
     */
    struct Parameters {
        int n;                  // Number of grid points per dimension
        double L;               // Domain half-length
        double T;               // Total simulation time
        int nt;                 // Number of time steps
        int snapshots;          // Number of snapshots to save
        std::optional<std::string> anisotropy_file; // File with anisotropy coefficients
        std::optional<std::string> focussing_file;  // File with focussing coefficients
        
        // Constructor with default values
        Parameters(int n_ = 200, double L_ = 10.0, double T_ = 10.0, int nt_ = 1000, int snapshots_ = 32)
            : n(n_), L(L_), T(T_), nt(nt_), snapshots(snapshots_) {}
    };

    /**
     * @brief Constructor
     * 
     * @param params The solver parameters
     */
    BaseSolver(const Parameters& params) 
        : params_(params), 
          dx_(2 * params.L / (params.n - 1)),
          dt_(params.T / params.nt),
          freq_(params.nt / params.snapshots) {
        
        // Calculate grid size based on dimension
        if constexpr (Dim == 2) {
            grid_size_ = params.n * params.n;
        } else if constexpr (Dim == 3) {
            grid_size_ = params.n * params.n * params.n;
        }
    }
    
    /**
     * @brief Virtual destructor
     */
    virtual ~BaseSolver() = default;
    
    /**
     * @brief Initialize the solver
     * 
     * This method should be called before starting the simulation.
     * It loads initial conditions, sets up the Laplacian matrix,
     * and prepares the solver for time stepping.
     * 
     * @param initial_u_file Path to the file with initial condition
     * @param initial_v_file Optional path to the file with initial velocity (for real waves)
     * @return true if initialization was successful, false otherwise
     */
    virtual bool initialize(const std::string& initial_u_file, 
                           const std::optional<std::string>& initial_v_file = std::nullopt) = 0;
    
    /**
     * @brief Perform a single time step
     * 
     * @return true if the step was successful, false otherwise
     */
    virtual bool step() = 0;
    
    /**
     * @brief Run the full simulation
     * 
     * @param trajectory_file Path to save the trajectory
     * @param velocity_file Optional path to save the velocity (for real waves)
     * @return true if the simulation was successful, false otherwise
     */
    virtual bool run(const std::string& trajectory_file,
                    const std::optional<std::string>& velocity_file = std::nullopt) = 0;
    
    /**
     * @brief Get the current state
     * 
     * @return const reference to the current state vector
     */
    virtual const Eigen::VectorX<Scalar_t>& get_state() const = 0;
    
    /**
     * @brief Get the current velocity (for real waves)
     * 
     * @return const reference to the current velocity vector
     */
    virtual const Eigen::VectorX<Scalar_t>& get_velocity() const = 0;
    
protected:
    Parameters params_;         // Solver parameters
    double dx_;                 // Grid spacing
    double dt_;                 // Time step
    int freq_;                  // Frequency of snapshots
    int grid_size_;             // Total number of grid points
    
    Eigen::VectorX<Scalar_t> u_;       // Current state
    Eigen::VectorX<double> c_;         // Anisotropy coefficients
    Eigen::VectorX<double> m_;         // Focussing coefficients
    Eigen::SparseMatrix<Scalar_t> L_;  // Laplacian matrix
    
    /**
     * @brief Load anisotropy and focussing coefficients
     * 
     * @return true if loading was successful, false otherwise
     */
    virtual bool load_coefficients() = 0;
    
    /**
     * @brief Build the Laplacian matrix
     * 
     * @return true if building was successful, false otherwise
     */
    virtual bool build_laplacian() = 0;
    
    /**
     * @brief Apply boundary conditions
     * 
     * @return true if applying was successful, false otherwise
     */
    virtual bool apply_boundary_conditions() = 0;
};

#endif // BASE_SOLVER_HPP
