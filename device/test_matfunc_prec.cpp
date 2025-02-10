#include "../eigen_krylov_real.hpp"
#include "../laplacians.hpp"
#include "matfunc_real.hpp"
#include <iomanip>

void test_solver_matfunc(const Eigen::SparseMatrix<double>& L, 
                        const Eigen::VectorXd& u0,
                        const double dt,
                        const uint32_t m = 10) {
    const uint32_t n = L.rows();
    const uint32_t nx = std::sqrt(n);
    const uint32_t ny = nx;
    
    int *d_row_ptr, *d_col_ind;
    double *d_values;
    cudaMalloc(&d_row_ptr, (n + 1) * sizeof(int));
    cudaMalloc(&d_col_ind, L.nonZeros() * sizeof(int));
    cudaMalloc(&d_values, L.nonZeros() * sizeof(double));

    cudaMemcpy(d_row_ptr, L.outerIndexPtr(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ind, L.innerIndexPtr(), L.nonZeros() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, L.valuePtr(), L.nonZeros() * sizeof(double), cudaMemcpyHostToDevice);

    MatrixFunctionApplicatorReal matfunc(d_row_ptr, d_col_ind, d_values, n, m, L.nonZeros());

    double *d_u, *d_result;
    cudaMalloc(&d_u, n * sizeof(double));
    cudaMalloc(&d_result, n * sizeof(double));
    cudaMemcpy(d_u, u0.data(), n * sizeof(double), cudaMemcpyHostToDevice);

    std::vector<double> result(n); 
    std::cout << std::scientific << std::setprecision(6) << "\n";
    
    {
        auto host_result = id_sqrt_multiply<double>(L, u0, dt, m);
        matfunc.apply(d_result, d_u, dt, MatrixFunctionApplicatorReal::FunctionType::ID_SQRT);
        cudaMemcpy(result.data(), d_result, n * sizeof(double), cudaMemcpyDeviceToHost);
        
        std::cout << "ID_SQRT comparison:\n";
        std::cout << "Host first 10: ";
        for(int i = 0; i < 10; ++i) std::cout << host_result[i] << " ";
        std::cout << "\nDevice first 10: ";
        for(int i = 0; i < 10; ++i) std::cout << result[i] << " ";
        std::cout << "\nDiff norm: " << (Eigen::Map<Eigen::VectorXd>(result.data(), n) - host_result).norm() << "\n\n";
    }

    {
        auto host_result = sinc2_sqrt_half<double>(L, u0, dt, m);
        matfunc.apply(d_result, d_u, dt/2, MatrixFunctionApplicatorReal::FunctionType::SINC2_SQRT);
        cudaMemcpy(result.data(), d_result, n * sizeof(double), cudaMemcpyDeviceToHost);
        
        std::cout << "SINC2_SQRT comparison:\n";
        std::cout << "Host first 10: ";
        for(int i = 0; i < 10; ++i) std::cout << host_result[i] << " ";
        std::cout << "\nDevice first 10: ";
        for(int i = 0; i < 10; ++i) std::cout << result[i] << " ";
        std::cout << "\nDiff norm: " << (Eigen::Map<Eigen::VectorXd>(result.data(), n) - host_result).norm() << "\n\n";
    }

    {
        auto host_result = cos_sqrt_multiply<double>(L, u0, dt, m);
        matfunc.apply(d_result, d_u, dt, MatrixFunctionApplicatorReal::FunctionType::COS_SQRT);
        cudaMemcpy(result.data(), d_result, n * sizeof(double), cudaMemcpyDeviceToHost);
        
        std::cout << "COS_SQRT comparison:\n";
        std::cout << "Host first 10: ";
        for(int i = 0; i < 10; ++i) std::cout << host_result[i] << " ";
        std::cout << "\nDevice first 10: ";
        for(int i = 0; i < 10; ++i) std::cout << result[i] << " ";
        std::cout << "\nDiff norm: " << (Eigen::Map<Eigen::VectorXd>(result.data(), n) - host_result).norm() << "\n\n";
    }

    cudaFree(d_row_ptr);
    cudaFree(d_col_ind);
    cudaFree(d_values);
    cudaFree(d_u);
    cudaFree(d_result);
}

int main() {
    const uint32_t nx = 128, ny = 128;
    const double Lx = 3., Ly = 3.;
    const double dx = 2 * Lx / (nx - 1), dy = 2 * Ly / (ny - 1);
    const double dt = 1e-2;
    
    Eigen::SparseMatrix<double> L = build_laplacian_noflux<double>(nx - 2, ny - 2, dx, dy); 
    auto f = [](double x, double y) {
        return 2. * std::atan(std::exp(3. - 5. * std::sqrt(x * x + y * y)));
    };
    
    Eigen::VectorXd u0(nx * ny);
    for(uint32_t i = 0; i < ny; ++i) {
        for(uint32_t j = 0; j < nx; ++j) {
            double x = -Lx + j * dx;
            double y = -Ly + i * dy;
            u0[i * nx + j] = f(x, y);
        }
    }
    
    test_solver_matfunc(L, u0, dt);
    return 0;
}
