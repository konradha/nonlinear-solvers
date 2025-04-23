#ifndef EXECUTION_CONTEXT_HPP
#define EXECUTION_CONTEXT_HPP

#include <Eigen/Sparse>
#include <memory>

/**
 * @brief Abstract base class for execution contexts.
 * 
 * This class defines the interface for different execution contexts
 * (host or device). It provides methods for matrix-vector operations
 * and other common operations used by the solvers.
 * 
 * @tparam Scalar_t The scalar type (double or std::complex<double>)
 */
template <typename Scalar_t>
class ExecutionContext {
public:
    /**
     * @brief Virtual destructor
     */
    virtual ~ExecutionContext() = default;
    
    /**
     * @brief Initialize the execution context
     * 
     * @param L The Laplacian matrix
     * @return true if initialization was successful, false otherwise
     */
    virtual bool initialize(const Eigen::SparseMatrix<Scalar_t>& L) = 0;
    
    /**
     * @brief Perform matrix-vector multiplication
     * 
     * @param x Input vector
     * @param y Output vector (y = L * x)
     * @return true if the operation was successful, false otherwise
     */
    virtual bool spmv(const Eigen::VectorX<Scalar_t>& x, Eigen::VectorX<Scalar_t>& y) = 0;
    
    /**
     * @brief Compute y = exp(tau * L) * x
     * 
     * @param x Input vector
     * @param y Output vector
     * @param tau Scalar factor
     * @return true if the operation was successful, false otherwise
     */
    virtual bool expm_multiply(const Eigen::VectorX<Scalar_t>& x, Eigen::VectorX<Scalar_t>& y, Scalar_t tau) = 0;
    
    /**
     * @brief Compute y = cos(sqrt(tau * L)) * x
     * 
     * @param x Input vector
     * @param y Output vector
     * @param tau Scalar factor
     * @return true if the operation was successful, false otherwise
     */
    virtual bool cos_sqrt_multiply(const Eigen::VectorX<Scalar_t>& x, Eigen::VectorX<Scalar_t>& y, Scalar_t tau) = 0;
    
    /**
     * @brief Compute y = sin(sqrt(tau * L)) / sqrt(tau * L) * x
     * 
     * @param x Input vector
     * @param y Output vector
     * @param tau Scalar factor
     * @return true if the operation was successful, false otherwise
     */
    virtual bool sinc_sqrt_multiply(const Eigen::VectorX<Scalar_t>& x, Eigen::VectorX<Scalar_t>& y, Scalar_t tau) = 0;
    
    /**
     * @brief Compute y = sin(sqrt(tau * L))^2 / (tau * L) * x
     * 
     * @param x Input vector
     * @param y Output vector
     * @param tau Scalar factor
     * @return true if the operation was successful, false otherwise
     */
    virtual bool sinc2_sqrt_half(const Eigen::VectorX<Scalar_t>& x, Eigen::VectorX<Scalar_t>& y, Scalar_t tau) = 0;
    
    /**
     * @brief Compute y = sqrt(tau * L) * x
     * 
     * @param x Input vector
     * @param y Output vector
     * @param tau Scalar factor
     * @return true if the operation was successful, false otherwise
     */
    virtual bool sqrt_multiply(const Eigen::VectorX<Scalar_t>& x, Eigen::VectorX<Scalar_t>& y, Scalar_t tau) = 0;
    
    /**
     * @brief Compute y = x / sqrt(tau * L)
     * 
     * @param x Input vector
     * @param y Output vector
     * @param tau Scalar factor
     * @return true if the operation was successful, false otherwise
     */
    virtual bool inv_sqrt_multiply(const Eigen::VectorX<Scalar_t>& x, Eigen::VectorX<Scalar_t>& y, Scalar_t tau) = 0;
    
    /**
     * @brief Create an execution context
     * 
     * Factory method to create an execution context based on the device type.
     * 
     * @param device_type The device type ("host" or "cuda")
     * @return std::unique_ptr<ExecutionContext<Scalar_t>> A unique pointer to the execution context
     */
    static std::unique_ptr<ExecutionContext<Scalar_t>> create(const std::string& device_type);
};

#endif // EXECUTION_CONTEXT_HPP
