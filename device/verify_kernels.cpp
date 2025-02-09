#include "pragmas.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>

// K1: f(t sqrt(λ_i))
__global__ void transform_eigenvals(double *out, const double *eigvals,
                                    const double t, const uint32_t m) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < m) {
    double x = t * sqrt(abs(eigvals[i]));
    out[i] = (abs(x) < 1e-8) ? 1.0 : pow(sin(x) / x, 2);
  }
}

// K2: Q f(λ) Q^T
__global__ void eigvec_transform(double *out, const double *Q,
                                 const double *f_lambda, const uint32_t m) {
  // let's take a single block
  int i = threadIdx.x;
  int j = threadIdx.y;
  if (i < m && j < m) {
    double sum = 0.0;
    for (int k = 0; k < m; k++) {
      sum += Q[k * m + i] * f_lambda[k] * Q[k * m + j];
    }
    out[i * m + j] = sum;
  }
}

// K3: β V(Q f(λ) Q^T)e_1
__global__ void final_multiply(double *result, const double *V,
                               const double *transform, const double beta,
                               const uint32_t n, const uint32_t m) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    double sum = 0.0;
    for (int j = 0; j < m; j++) {
      double v_sum = 0.0;
      for (int k = 0; k < m; k++) {
        v_sum += V[k * n + i] * transform[k * m + j];
      }
      sum = (j == 0) ? v_sum : sum;
    }
    result[i] = beta * sum;
  }
}

class KernelVerifier {
private:
  uint32_t n_, m_;
  double *d_V, *d_eigvals, *d_eigvecs, *d_temp, *d_sinc_result, *d_final;
  double t_ = 0.5;
  dim3 grid_1d_, block_1d_, grid_2d_, block_2d_;

  Eigen::VectorXd verify_transform_eigenvals(const Eigen::VectorXd &eigenvals) {
    return (t_ / 2. * eigenvals.array().abs().sqrt())
        .unaryExpr(
            [](double x) { return std::abs(x) < 1e-8 ? 1.0 : std::sin(x) / x; })
        .square();
  }

  Eigen::MatrixXd verify_eigvec_transform(const Eigen::MatrixXd &Q,
                                          const Eigen::VectorXd &f_lambda) {
    return Q * f_lambda.asDiagonal() * Q.transpose();
  }

  Eigen::VectorXd verify_final_multiply(const Eigen::MatrixXd &V,
                                        const Eigen::MatrixXd &transform) {
    Eigen::VectorXd e1 = Eigen::VectorXd::Zero(m_);
    e1(0) = 1.0;
    return V * transform * e1;
  }

public:
  KernelVerifier(uint32_t n, uint32_t m) : n_(n), m_(m) {
    block_1d_ = dim3(256);
    grid_1d_ = dim3((n + block_1d_.x - 1) / block_1d_.x);
    block_2d_ = dim3(16, 16);
    grid_2d_ = dim3((m + block_2d_.x - 1) / block_2d_.x,
                    (m + block_2d_.y - 1) / block_2d_.y);

    cudaMalloc(&d_V, n_ * m_ * sizeof(double));
    cudaMalloc(&d_eigvals, m_ * sizeof(double));
    cudaMalloc(&d_eigvecs, m_ * m_ * sizeof(double));
    cudaMalloc(&d_temp, m_ * sizeof(double));
    cudaMalloc(&d_sinc_result, m_ * m_ * sizeof(double));
    cudaMalloc(&d_final, n_ * sizeof(double));
  }

  void run_verification() {
    Eigen::MatrixXd T = Eigen::MatrixXd::Random(m_, m_);
    T = (T + T.transpose()) / 2;
    Eigen::MatrixXd V = Eigen::MatrixXd::Random(n_, m_);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(T);
    cudaMemcpy(d_V, V.data(), n_ * m_ * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_eigvals, es.eigenvalues().data(), m_ * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_eigvecs, es.eigenvectors().data(), m_ * m_ * sizeof(double),
               cudaMemcpyHostToDevice);

    transform_eigenvals<<<grid_1d_, block_1d_>>>(d_temp, d_eigvals, t_ / 2, m_);
    Eigen::VectorXd eigen_temp = verify_transform_eigenvals(es.eigenvalues());
    Eigen::VectorXd cuda_temp(m_);
    cudaMemcpy(cuda_temp.data(), d_temp, m_ * sizeof(double),
               cudaMemcpyDeviceToHost);
    std::cout << "Stage 1 diff - L1: " << (eigen_temp - cuda_temp).lpNorm<1>()
              << " L2: " << (eigen_temp - cuda_temp).lpNorm<2>() << std::endl;

    eigvec_transform<<<grid_2d_, block_2d_>>>(d_sinc_result, d_eigvecs, d_temp,
                                              m_);
    Eigen::MatrixXd eigen_sinc =
        verify_eigvec_transform(es.eigenvectors(), eigen_temp);
    Eigen::MatrixXd cuda_sinc(m_, m_);
    cudaMemcpy(cuda_sinc.data(), d_sinc_result, m_ * m_ * sizeof(double),
               cudaMemcpyDeviceToHost);
    std::cout << "Stage 2 diff - L1: " << (eigen_sinc - cuda_sinc).lpNorm<1>()
              << " L2: " << (eigen_sinc - cuda_sinc).lpNorm<2>() << std::endl;

    final_multiply<<<grid_1d_, block_1d_>>>(d_final, d_V, d_sinc_result, 1.0,
                                            n_, m_);
    Eigen::VectorXd eigen_final = verify_final_multiply(V, eigen_sinc);
    Eigen::VectorXd cuda_final(n_);
    cudaMemcpy(cuda_final.data(), d_final, n_ * sizeof(double),
               cudaMemcpyDeviceToHost);
    std::cout << "Stage 3 diff - L1: " << (eigen_final - cuda_final).lpNorm<1>()
              << " L2: " << (eigen_final - cuda_final).lpNorm<2>() << std::endl;
  }

  ~KernelVerifier() {
    cudaFree(d_V);
    cudaFree(d_eigvals);
    cudaFree(d_eigvecs);
    cudaFree(d_temp);
    cudaFree(d_sinc_result);
    cudaFree(d_final);
  }
};

int main() {
  std::vector<std::pair<uint32_t, uint32_t>> sizes = {
      {1000, 10}, {10000, 10}, {100000, 10}};

  for (const auto &[n, m] : sizes) {
    std::cout << "\nTesting n=" << n << ", m=" << m << std::endl;
    KernelVerifier verifier(n, m);
    verifier.run_verification();
  }
  return 0;
}
