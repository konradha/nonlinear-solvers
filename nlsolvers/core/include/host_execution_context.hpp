#ifndef HOST_EXECUTION_CONTEXT_HPP
#define HOST_EXECUTION_CONTEXT_HPP

#include "execution_context.hpp"
#include <Eigen/Sparse>

/**
 * @brief Host execution context implementation.
 * 
 * This class implements the execution context interface for host (CPU) execution.
 * It uses the Eigen library for matrix operations.
 * 
 * @tparam Scalar_t The scalar type (double or std::complex<double>)
 */
template <typename Scalar_t>
class HostExecutionContext : public ExecutionContext<Scalar_t> {
public:
    /**
     * @brief Constructor
     */
    HostExecutionContext() = default;
    
    /**
     * @brief Destructor
     */
    ~HostExecutionContext() override = default;
    
    /**
     * @brief Initialize the execution context
     * 
     * @param L The Laplacian matrix
     * @return true if initialization was successful, false otherwise
     */
    bool initialize(const Eigen::SparseMatrix<Scalar_t>& L) override {
        L_ = L;
        return true;
    }
    
    /**
     * @brief Perform matrix-vector multiplication
     * 
     * @param x Input vector
     * @param y Output vector (y = L * x)
     * @return true if the operation was successful, false otherwise
     */
    bool spmv(const Eigen::VectorX<Scalar_t>& x, Eigen::VectorX<Scalar_t>& y) override {
        y = L_ * x;
        return true;
    }
    
    /**
     * @brief Compute y = exp(tau * L) * x
     * 
     * @param x Input vector
     * @param y Output vector
     * @param tau Scalar factor
     * @return true if the operation was successful, false otherwise
     */
    bool expm_multiply(const Eigen::VectorX<Scalar_t>& x, Eigen::VectorX<Scalar_t>& y, Scalar_t tau) override {
        // Use Krylov subspace method to compute the matrix exponential
        y = expm_multiply_krylov(L_, x, tau);
        return true;
    }
    
    /**
     * @brief Compute y = cos(sqrt(tau * L)) * x
     * 
     * @param x Input vector
     * @param y Output vector
     * @param tau Scalar factor
     * @return true if the operation was successful, false otherwise
     */
    bool cos_sqrt_multiply(const Eigen::VectorX<Scalar_t>& x, Eigen::VectorX<Scalar_t>& y, Scalar_t tau) override {
        // Use Krylov subspace method to compute the cosine function
        y = cos_sqrt_multiply_krylov(L_, x, tau);
        return true;
    }
    
    /**
     * @brief Compute y = sin(sqrt(tau * L)) / sqrt(tau * L) * x
     * 
     * @param x Input vector
     * @param y Output vector
     * @param tau Scalar factor
     * @return true if the operation was successful, false otherwise
     */
    bool sinc_sqrt_multiply(const Eigen::VectorX<Scalar_t>& x, Eigen::VectorX<Scalar_t>& y, Scalar_t tau) override {
        // Use Krylov subspace method to compute the sinc function
        y = sinc_sqrt_multiply_krylov(L_, x, tau);
        return true;
    }
    
    /**
     * @brief Compute y = sin(sqrt(tau * L))^2 / (tau * L) * x
     * 
     * @param x Input vector
     * @param y Output vector
     * @param tau Scalar factor
     * @return true if the operation was successful, false otherwise
     */
    bool sinc2_sqrt_half(const Eigen::VectorX<Scalar_t>& x, Eigen::VectorX<Scalar_t>& y, Scalar_t tau) override {
        // Use Krylov subspace method to compute the sinc^2 function
        y = sinc2_sqrt_half_krylov(L_, x, tau);
        return true;
    }
    
    /**
     * @brief Compute y = sqrt(tau * L) * x
     * 
     * @param x Input vector
     * @param y Output vector
     * @param tau Scalar factor
     * @return true if the operation was successful, false otherwise
     */
    bool sqrt_multiply(const Eigen::VectorX<Scalar_t>& x, Eigen::VectorX<Scalar_t>& y, Scalar_t tau) override {
        // Use Krylov subspace method to compute the square root function
        y = sqrt_multiply_krylov(L_, x, tau);
        return true;
    }
    
    /**
     * @brief Compute y = x / sqrt(tau * L)
     * 
     * @param x Input vector
     * @param y Output vector
     * @param tau Scalar factor
     * @return true if the operation was successful, false otherwise
     */
    bool inv_sqrt_multiply(const Eigen::VectorX<Scalar_t>& x, Eigen::VectorX<Scalar_t>& y, Scalar_t tau) override {
        // Use Krylov subspace method to compute the inverse square root function
        y = inv_sqrt_multiply_krylov(L_, x, tau);
        return true;
    }
    
private:
    Eigen::SparseMatrix<Scalar_t> L_;  // Laplacian matrix
    
    // Krylov subspace methods for matrix functions
    // These are implemented in eigen_krylov_real.hpp and eigen_krylov_complex.hpp
    
    Eigen::VectorX<Scalar_t> expm_multiply_krylov(const Eigen::SparseMatrix<Scalar_t>& A, 
                                                 const Eigen::VectorX<Scalar_t>& v, 
                                                 Scalar_t t);
    
    Eigen::VectorX<Scalar_t> cos_sqrt_multiply_krylov(const Eigen::SparseMatrix<Scalar_t>& A, 
                                                     const Eigen::VectorX<Scalar_t>& v, 
                                                     Scalar_t t);
    
    Eigen::VectorX<Scalar_t> sinc_sqrt_multiply_krylov(const Eigen::SparseMatrix<Scalar_t>& A, 
                                                      const Eigen::VectorX<Scalar_t>& v, 
                                                      Scalar_t t);
    
    Eigen::VectorX<Scalar_t> sinc2_sqrt_half_krylov(const Eigen::SparseMatrix<Scalar_t>& A, 
                                                   const Eigen::VectorX<Scalar_t>& v, 
                                                   Scalar_t t);
    
    Eigen::VectorX<Scalar_t> sqrt_multiply_krylov(const Eigen::SparseMatrix<Scalar_t>& A, 
                                                 const Eigen::VectorX<Scalar_t>& v, 
                                                 Scalar_t t);
    
    Eigen::VectorX<Scalar_t> inv_sqrt_multiply_krylov(const Eigen::SparseMatrix<Scalar_t>& A, 
                                                     const Eigen::VectorX<Scalar_t>& v, 
                                                     Scalar_t t);
};

// Template specialization for double
template <>
Eigen::VectorXd HostExecutionContext<double>::expm_multiply_krylov(const Eigen::SparseMatrix<double>& A, 
                                                                  const Eigen::VectorXd& v, 
                                                                  double t) {
    // Implementation will use the existing expm_multiply function from eigen_krylov_real.hpp
    return expm_multiply(A, v, t);
}

template <>
Eigen::VectorXd HostExecutionContext<double>::cos_sqrt_multiply_krylov(const Eigen::SparseMatrix<double>& A, 
                                                                      const Eigen::VectorXd& v, 
                                                                      double t) {
    // Implementation will use the existing cos_sqrt_multiply function from eigen_krylov_real.hpp
    return cos_sqrt_multiply(A, v, t);
}

template <>
Eigen::VectorXd HostExecutionContext<double>::sinc_sqrt_multiply_krylov(const Eigen::SparseMatrix<double>& A, 
                                                                       const Eigen::VectorXd& v, 
                                                                       double t) {
    // Implementation will use the existing sinc_sqrt_multiply function from eigen_krylov_real.hpp
    return sinc_sqrt_multiply(A, v, t);
}

template <>
Eigen::VectorXd HostExecutionContext<double>::sinc2_sqrt_half_krylov(const Eigen::SparseMatrix<double>& A, 
                                                                    const Eigen::VectorXd& v, 
                                                                    double t) {
    // Implementation will use the existing sinc2_sqrt_half function from eigen_krylov_real.hpp
    return sinc2_sqrt_half(A, v, t);
}

template <>
Eigen::VectorXd HostExecutionContext<double>::sqrt_multiply_krylov(const Eigen::SparseMatrix<double>& A, 
                                                                  const Eigen::VectorXd& v, 
                                                                  double t) {
    // Implementation will use the existing sqrt_multiply function from eigen_krylov_real.hpp
    return sqrt_multiply(A, v, t);
}

template <>
Eigen::VectorXd HostExecutionContext<double>::inv_sqrt_multiply_krylov(const Eigen::SparseMatrix<double>& A, 
                                                                      const Eigen::VectorXd& v, 
                                                                      double t) {
    // Implementation will use the existing inv_sqrt_multiply function from eigen_krylov_real.hpp
    return inv_sqrt_multiply(A, v, t);
}

// Template specialization for std::complex<double>
template <>
Eigen::VectorXcd HostExecutionContext<std::complex<double>>::expm_multiply_krylov(const Eigen::SparseMatrix<std::complex<double>>& A, 
                                                                                const Eigen::VectorXcd& v, 
                                                                                std::complex<double> t) {
    // Implementation will use the existing expm_multiply function from eigen_krylov_complex.hpp
    return expm_multiply(A, v, t);
}

template <>
Eigen::VectorXcd HostExecutionContext<std::complex<double>>::cos_sqrt_multiply_krylov(const Eigen::SparseMatrix<std::complex<double>>& A, 
                                                                                    const Eigen::VectorXcd& v, 
                                                                                    std::complex<double> t) {
    // Implementation will use the existing cos_sqrt_multiply function from eigen_krylov_complex.hpp
    return cos_sqrt_multiply(A, v, t);
}

template <>
Eigen::VectorXcd HostExecutionContext<std::complex<double>>::sinc_sqrt_multiply_krylov(const Eigen::SparseMatrix<std::complex<double>>& A, 
                                                                                     const Eigen::VectorXcd& v, 
                                                                                     std::complex<double> t) {
    // Implementation will use the existing sinc_sqrt_multiply function from eigen_krylov_complex.hpp
    return sinc_sqrt_multiply(A, v, t);
}

template <>
Eigen::VectorXcd HostExecutionContext<std::complex<double>>::sinc2_sqrt_half_krylov(const Eigen::SparseMatrix<std::complex<double>>& A, 
                                                                                  const Eigen::VectorXcd& v, 
                                                                                  std::complex<double> t) {
    // Implementation will use the existing sinc2_sqrt_half function from eigen_krylov_complex.hpp
    return sinc2_sqrt_half(A, v, t);
}

template <>
Eigen::VectorXcd HostExecutionContext<std::complex<double>>::sqrt_multiply_krylov(const Eigen::SparseMatrix<std::complex<double>>& A, 
                                                                                const Eigen::VectorXcd& v, 
                                                                                std::complex<double> t) {
    // Implementation will use the existing sqrt_multiply function from eigen_krylov_complex.hpp
    return sqrt_multiply(A, v, t);
}

template <>
Eigen::VectorXcd HostExecutionContext<std::complex<double>>::inv_sqrt_multiply_krylov(const Eigen::SparseMatrix<std::complex<double>>& A, 
                                                                                    const Eigen::VectorXcd& v, 
                                                                                    std::complex<double> t) {
    // Implementation will use the existing inv_sqrt_multiply function from eigen_krylov_complex.hpp
    return inv_sqrt_multiply(A, v, t);
}

// Factory method implementation
template <typename Scalar_t>
std::unique_ptr<ExecutionContext<Scalar_t>> ExecutionContext<Scalar_t>::create(const std::string& device_type) {
    if (device_type == "host") {
        return std::make_unique<HostExecutionContext<Scalar_t>>();
    } else if (device_type == "cuda") {
        // Device execution context will be implemented later
        // return std::make_unique<DeviceExecutionContext<Scalar_t>>();
        throw std::runtime_error("CUDA execution context not implemented yet");
    } else {
        throw std::runtime_error("Unknown device type: " + device_type);
    }
}

#endif // HOST_EXECUTION_CONTEXT_HPP
