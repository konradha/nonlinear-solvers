// nvcc -O2 --fmad=true --restrict -std=c++17 --expt-relaxed-constexpr -diag-suppress 20012 -diag-suppress 20011 -lcusparse -lcublas -lcusolver -Xcompiler -fopenmp -x cu --extended-lambda -arch=sm_70 nlse_driver_omp.cpp -o to_nlse_driver_par
#include "../nlse_solver.hpp"
#include "../util.hpp"
#include "nlse_solver_dev.hpp"
#include <chrono>
#include <iomanip>
#include <omp.h>
#include <vector>
#include <future>

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
    int n_threads;

    if(const char* env_cpus = std::getenv("SLURM_CPUS_PER_TASK")) {
        n_threads = std::atoi(env_cpus);
    } else {
        n_threads = omp_get_max_threads();
    }
    omp_set_num_threads(n_threads);

    const uint32_t nx = 256, ny = 256;
    const f_ty Lx = 10., Ly = 10.;
    const f_ty dx = 2 * Lx / (nx - 1), dy = 2 * Ly / (ny - 1);
    const f_ty T = 1.5;
    const uint32_t nt = 500;
    const uint32_t num_snapshots = 100;
    const auto freq = nt / num_snapshots;
    const auto dt = T / nt;
    c_ty dti(0, dt);

    const f_ty sigma = .2;
    const f_ty A = 1.;
    const f_ty r = 3.;
    const f_ty k0 = .5;

    std::vector<Eigen::VectorX<c_ty>> u0_vectors(n_threads);
    std::vector<Eigen::VectorX<c_ty>> u_save_gpu(n_threads);
    std::vector<double> evolution_times(n_threads);
    std::vector<double> io_times(n_threads);

    const Eigen::SparseMatrix<c_ty> L = 
        build_laplacian_noflux<c_ty>(nx - 2, ny - 2, dx, dy);

    #pragma omp parallel
    {
        const int thread_id = omp_get_thread_num();
        const f_ty angle = 2. * M_PI * thread_id / n_threads;
        const f_ty cos_a = std::cos(angle);
        const f_ty sin_a = std::sin(angle);
        
        f_ty x1 = r * cos_a;
        f_ty y1 = r * sin_a;
        f_ty x2 = -x1;
        f_ty y2 = -y1;
        
        f_ty kx = -x1/r * k0;
        f_ty ky = -y1/r * k0;

        auto f_collision = [&](c_ty x, c_ty y) {
            auto un1 = A * std::exp(-((x - x1) * (x - x1) + 
                                    (y - y1) * (y - y1)) /
                                   4. / sigma / sigma) *
                      std::exp(c_ty(0, 1.) * (kx * x + ky * y));
            auto un2 = A * std::exp(-((x - x2) * (x - x2) + 
                                    (y - y2) * (y - y2)) /
                                   4. / sigma / sigma) *
                      std::exp(c_ty(0, -1.) * (kx * x + ky * y));
            return un1 + un2;
        };

        u0_vectors[thread_id] = 
            apply_function_uniform<c_ty>(-Lx, Lx, nx, -Ly, Ly, ny, f_collision);
        
        auto get_norm = [&](const Eigen::VectorX<c_ty> &x) {
            return std::sqrt((x.conjugate().array() * x.array() * dx * dy).sum());
        };
        const auto Norm = get_norm(u0_vectors[thread_id]);
        u0_vectors[thread_id] = u0_vectors[thread_id] / Norm;
        
        u_save_gpu[thread_id].resize(num_snapshots * nx * ny);
    }

    #pragma omp parallel
    {
        const int thread_id = omp_get_thread_num();
        NLSESolverDevice::Parameters params(num_snapshots, freq);
        NLSESolverDevice solver(L, u0_vectors[thread_id].data(), params);

        auto start = std::chrono::high_resolution_clock::now();

        for (uint32_t t = 1; t < nt; ++t) {
            solver.step(dti, t);
        }

        solver.transfer_snapshots(u_save_gpu[thread_id].data());

        auto end = std::chrono::high_resolution_clock::now();
        evolution_times[thread_id] = 
            std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                .count();
    }

    const std::vector<uint32_t> shape = {num_snapshots, nx, ny};
    
    

#pragma omp parallel
    {
        const int thread_id = omp_get_thread_num();
        const auto fname = "evolution_nlse_device_" + std::to_string(thread_id) + ".npy";
        
        auto io_start = std::chrono::high_resolution_clock::now();
        save_to_npy(fname, u_save_gpu[thread_id], shape);
        auto io_end = std::chrono::high_resolution_clock::now();
        
        io_times[thread_id] = 
            std::chrono::duration_cast<std::chrono::microseconds>(io_end - io_start)
                .count();
    }

    std::cout << std::scientific << std::setprecision(4);
    std::cout << "\nComputation times:\n";
    for(uint32_t i = 0; i < n_threads; ++i) {
        std::cout << "Evolution " << i << ":\n";
        std::cout << "  Compute time: " << evolution_times[i] << " us\n";
        std::cout << "  I/O time:     " << io_times[i] << " us\n";
        std::cout << "  I/O overhead: " << (io_times[i] / evolution_times[i]) * 100 << "%\n";
    }
    
    double avg_compute_time = 0.0;
    double avg_io_time = 0.0;
    for(uint32_t i = 0; i < n_threads; ++i) {
        avg_compute_time += evolution_times[i];
        avg_io_time += io_times[i];
    }
    avg_compute_time /= n_threads;
    avg_io_time /= n_threads;
    
    std::cout << "\nAverages:\n";
    std::cout << "Average compute time: " << avg_compute_time << " us\n";
    std::cout << "Average I/O time:     " << avg_io_time << " us\n";
    std::cout << "Average I/O overhead: " << (avg_io_time / avg_compute_time) * 100 << "%\n";

    return 0;
}
