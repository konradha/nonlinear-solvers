#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <random>
#include <iomanip>
#include <iostream>

// V x K multiplication (n x m) * (m x m)
__global__ void matrix_multiply_VK(const double* V, const double* K, double* result,
                                 const uint32_t n, const uint32_t m) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < m) {
        double sum = 0.0;
        for (int k = 0; k < m; k++) {
            sum += V[k * n + row] * K[col * m + k];
        }
        result[col * n + row] = sum;
    }
}

// Q * diag(f(D)) * Q^T
__global__ void matrix_multiply_QDQ(const double* Q, const double* D, double* result,
                                  const uint32_t m) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < m) {
        double sum = 0.0;
        for (int k = 0; k < m; k++) {
            sum += Q[k * m + row] * D[k] * Q[k * m + col];
        }
        result[col * m + row] = sum;
    }
}

// beta * X[0, :]
__global__ void scale_first_col(const double* X, double* result,
                                 const double beta, const uint32_t n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx < n) {
        result[idx] = beta * X[idx];
    }
}

void test_matrix_operations(uint32_t n, uint32_t m, uint32_t num_trials = 10) {
    std::mt19937 gen(42);
    std::normal_distribution<double> dist(0.0, 1.0);
    
    double *d_V, *d_K, *d_Q, *d_D, *d_result, *d_temp;
    cudaMalloc(&d_V, n * m * sizeof(double));
    cudaMalloc(&d_K, m * m * sizeof(double));
    cudaMalloc(&d_Q, m * m * sizeof(double));
    cudaMalloc(&d_D, m * sizeof(double));
    cudaMalloc(&d_result, n * m * sizeof(double));
    cudaMalloc(&d_temp, m * m * sizeof(double));
    
    dim3 block_2d(16, 16);
    dim3 grid_VK((m + block_2d.x - 1) / block_2d.x,
                 (n + block_2d.y - 1) / block_2d.y);
    dim3 grid_QDQ((m + block_2d.x - 1) / block_2d.x,
                  (m + block_2d.y - 1) / block_2d.y);
    dim3 block_1d(256);
    dim3 grid_1d((n + block_1d.x - 1) / block_1d.x);
    
    for (uint32_t trial = 0; trial < num_trials; ++trial) {
        Eigen::MatrixXd V = Eigen::MatrixXd::NullaryExpr(n, m, 
            [&]() { return dist(gen); });
        Eigen::MatrixXd K = Eigen::MatrixXd::NullaryExpr(m, m, 
            [&]() { return dist(gen); });
        Eigen::MatrixXd Q = Eigen::MatrixXd::NullaryExpr(m, m, 
            [&]() { return dist(gen); });
        Eigen::VectorXd D = Eigen::VectorXd::NullaryExpr(m, 
            [&]() { return dist(gen); });
        double beta = dist(gen);
        
        // V x K
        cudaMemcpy(d_V, V.data(), n * m * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_K, K.data(), m * m * sizeof(double), cudaMemcpyHostToDevice);
        
        matrix_multiply_VK<<<grid_VK, block_2d>>>(d_V, d_K, d_result, n, m);
        
        Eigen::MatrixXd result_VK(n, m);
        cudaMemcpy(result_VK.data(), d_result, n * m * sizeof(double), 
                  cudaMemcpyDeviceToHost);
        
        Eigen::MatrixXd ref_VK = V * K;
        std::cout << "Trial " << trial << " V x K error: " 
                  << (result_VK - ref_VK).norm() << "\n";
        
        // Q * diag(D) * Q^T
        cudaMemcpy(d_Q, Q.data(), m * m * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_D, D.data(), m * sizeof(double), cudaMemcpyHostToDevice);
        
        matrix_multiply_QDQ<<<grid_QDQ, block_2d>>>(d_Q, d_D, d_temp, m);
        
        Eigen::MatrixXd result_QDQ(m, m);
        cudaMemcpy(result_QDQ.data(), d_temp, m * m * sizeof(double), 
                  cudaMemcpyDeviceToHost);
        
        Eigen::MatrixXd ref_QDQ = Q * D.asDiagonal() * Q.transpose();
        std::cout << "Trial " << trial << " QDQ^T error: " 
                  << (result_QDQ - ref_QDQ).norm() << "\n";
        
        // beta * X[0, :]
        scale_first_col<<<grid_1d, block_1d>>>(d_V, d_result, beta, n);
        
        Eigen::VectorXd result_scale(n);
        cudaMemcpy(result_scale.data(), d_result, n * sizeof(double), 
                  cudaMemcpyDeviceToHost);
        
        Eigen::VectorXd ref_scale = beta * V.col(0);
        std::cout << "Trial " << trial << " beta * X[0,:] error: " 
                  << (result_scale - ref_scale).norm() << "\n\n";
    }
    
    cudaFree(d_V);
    cudaFree(d_K);
    cudaFree(d_Q);
    cudaFree(d_D);
    cudaFree(d_result);
    cudaFree(d_temp);
}

int main() {
    std::vector<std::pair<uint32_t, uint32_t>> test_sizes = {
        {128, 10}, {256, 10}, {512, 10}, {1024, 10}
    };
    
    for (const auto& [n, m] : test_sizes) {
        std::cout << "\nTesting n = " << n << ", m = " << m << "\n";
        test_matrix_operations(n, m);
    }
    
    return 0;
}
