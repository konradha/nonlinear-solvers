#include "../nlse_solver.hpp"
#include "../util.hpp"
#include "nlse_solver_dev.hpp"
#include <chrono>
#include <iomanip>

template <typename Float, typename F>
Eigen::VectorX<Float> apply_function_uniform(Float x_min, Float x_max,
                                           uint32_t nx, Float y_min,
                                           Float y_max, uint32_t ny, F f) {
    Eigen::VectorX<Float> x = Eigen::VectorX<Float>::LinSpaced(nx, x_min, x_max);
    Eigen::VectorX<Float> y = Eigen::VectorX<Float>::LinSpaced(ny, y_min, y_max);
    Eigen::VectorX<Float> u(nx * ny);
    for (uint32_t i = 0; i < ny; ++i) {
        for (uint32_t j = 0; j < nx; ++j) {
            u[i * nx + j] = f(x[i], y[j]);
        }
    }
    return u;
}

int main() {
    using f_ty = double;
    using c_ty = std::complex<f_ty>;

    const uint32_t nx = 256, ny = 256;
    const f_ty Lx = 10., Ly = 10.;
    const f_ty dx = 2 * Lx / (nx - 1), dy = 2 * Ly / (ny - 1);
    const f_ty T = 1.5;
    const uint32_t nt = 500;
    const uint32_t num_snapshots = 100;
    const auto freq = nt / num_snapshots;
    const auto dt = T / nt;
    c_ty dti(0, dt);

    f_ty k0 = .5;
    f_ty theta = 3.14159265358979323846 * .25;
    f_ty kx = k0 * std::cos(theta);
    f_ty ky = k0 * std::sin(theta);
    f_ty sigma = .2;
    f_ty A = 1.;

    auto f_collision = [&](c_ty x, c_ty y) {
        auto un1 = A * std::exp(-((x - 3.) * (x - 3.) + (y - 3.) * (y - 3.)) /
                               4. / sigma / sigma) * 
                   std::exp(c_ty(0, -1.) * (x + y));
        auto un2 = A * std::exp(-((x + 3.) * (x + 3.) + (y + 3.) * (y + 3.)) /
                               4. / sigma / sigma) * 
                   std::exp(c_ty(0, 1.) * (x + y));
        return un1 + un2;
    };

    Eigen::VectorX<c_ty> u0 = 
        apply_function_uniform<c_ty>(-Lx, Lx, nx, -Ly, Ly, ny, f_collision);
    auto get_norm = [&](const Eigen::VectorX<c_ty> &x) {
        return std::sqrt((x.array().abs2() * dx * dy).sum());
    };
    const auto Norm = get_norm(u0);
    u0 = u0 / Norm;

    Eigen::VectorX<c_ty> u_save_cpu(num_snapshots * nx * ny);
    Eigen::VectorX<c_ty> u_save_gpu(num_snapshots * nx * ny);

    const Eigen::SparseMatrix<c_ty> L = 
        build_laplacian_noflux<c_ty>(nx - 2, ny - 2, dx, dy);

    auto cpu_start = std::chrono::high_resolution_clock::now();
    {
        Eigen::VectorX<c_ty> u = u0;
        Eigen::VectorX<c_ty> buf = u0;
        Eigen::VectorX<c_ty> rho_buf = u0;
        
        Eigen::Map<Eigen::Matrix<c_ty, -1, -1, Eigen::RowMajor>> u_save_mat(
            u_save_cpu.data(), num_snapshots, nx * ny);
        
        u_save_mat.row(0) = u0.transpose();

        for (uint32_t i = 1; i < nt; ++i) {
            NLSESolver::step<c_ty>(buf, rho_buf, u, L, dti);
            if (i % freq == 0) {
                u_save_mat.row(i / freq) = u.transpose();
            }
        }
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_time = 
        std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start)
            .count();

    auto gpu_start = std::chrono::high_resolution_clock::now();
    {
        NLSESolverDevice::Parameters params(num_snapshots, freq);
        NLSESolverDevice solver(L, u0.data(), params);

        for (uint32_t i = 1; i < nt; ++i) {
            solver.step(dti, i);
        }
        solver.transfer_snapshots(u_save_gpu.data());
    }
    auto gpu_end = std::chrono::high_resolution_clock::now();
    auto gpu_time = 
        std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start)
            .count();

    Eigen::VectorXcd u_diff = u_save_cpu - u_save_gpu;

    std::cout << std::scientific << std::setprecision(4);

    std::cout << "L2 diff: " << u_diff.norm() << "\n";
    std::cout << "host: " << cpu_time << " us\n";
    std::cout << "dev:  " << gpu_time << " us\n";
    std::cout << "s: " << static_cast<double>(cpu_time) / gpu_time << "x\n";

    const std::vector<uint32_t> shape = {num_snapshots, nx, ny};
    const auto fname_u_cpu = "evolution_nlse_host.npy";
    const auto fname_u_gpu = "evolution_nlse_device.npy";

    save_to_npy(fname_u_cpu, u_save_cpu, shape);
    save_to_npy(fname_u_gpu, u_save_gpu, shape);

    return 0;
}
