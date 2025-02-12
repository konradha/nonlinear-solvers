#include "../laplacians.hpp"
#include "../sg_solver.hpp"
#include "../util.hpp"
#include "sg_solver_dev.hpp"
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
  const uint32_t nx = 256, ny = 256;
  const f_ty Lx = 3., Ly = 3.;
  const f_ty dx = 2 * Lx / (nx - 1), dy = 2 * Ly / (ny - 1);
  const f_ty T = 5.;
  const uint32_t nt = 500;
  const uint32_t num_snapshots = 100;
  const auto freq = nt / num_snapshots;
  const auto dt = T / nt;

  auto f = [](f_ty x, f_ty y) {
    return 2. * std::atan(std::exp(3. - 5. * std::sqrt(x * x + y * y)));
  };
  auto zero = [](f_ty x, f_ty y) { return 0.; };
  auto one = [](f_ty x, f_ty y) { return 1.; };
  auto neg = [](f_ty x, f_ty y) { return -1.; };

  Eigen::VectorX<f_ty> u0 =
      apply_function_uniform<f_ty>(-Lx, Lx, nx, -Ly, Ly, ny, f);
  Eigen::VectorX<f_ty> v0 =
      apply_function_uniform<f_ty>(-Lx, Lx, nx, -Ly, Ly, ny, zero);
  Eigen::VectorX<f_ty> m =
      apply_function_uniform<f_ty>(-Lx, Lx, nx, -Ly, Ly, ny, neg);
  Eigen::VectorX<f_ty> c =
      apply_function_uniform<f_ty>(-Lx, Lx, nx, -Ly, Ly, ny, one);

  const Eigen::SparseMatrix<f_ty> L =
      build_laplacian_noflux<f_ty>(nx - 2, ny - 2, dx, dy);

  Eigen::VectorX<f_ty> u_save_cpu(num_snapshots * nx * ny);
  Eigen::VectorX<f_ty> v_save_cpu(num_snapshots * nx * ny);
  Eigen::VectorX<f_ty> u_save_gpu(num_snapshots * nx * ny);
  Eigen::VectorX<f_ty> v_save_gpu(num_snapshots * nx * ny);
  Eigen::VectorX<f_ty> e_save_gpu(num_snapshots);

  auto cpu_start = std::chrono::high_resolution_clock::now();
  {
    Eigen::VectorX<f_ty> u = u0;
    Eigen::VectorX<f_ty> v = v0;
    Eigen::VectorX<f_ty> buf = v0;
    Eigen::VectorX<f_ty> u_past = u0 - dt * v0;
    Eigen::Map<Eigen::Matrix<f_ty, -1, -1, Eigen::RowMajor>> u_save_mat(
        u_save_cpu.data(), num_snapshots, nx * ny);
    Eigen::Map<Eigen::Matrix<f_ty, -1, -1, Eigen::RowMajor>> v_save_mat(
        v_save_cpu.data(), num_snapshots, nx * ny);

    u_save_mat.row(0) = u0.transpose();
    v_save_mat.row(0) = v0.transpose();

    for (uint32_t i = 1; i < nt; ++i) {
      SGESolver::step<f_ty>(u, u_past, buf, L, c, m, dt);
      v = (u - u_past) / dt;
      if (i % freq == 0) {
        u_save_mat.row(i / freq) = u.transpose();
        v_save_mat.row(i / freq) = v.transpose();
      }
    }
  }
  auto cpu_end = std::chrono::high_resolution_clock::now();
  auto cpu_time =
      std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start)
          .count();

  int *d_row_ptr, *d_col_ind;
  double *d_values;

  cudaMalloc(&d_row_ptr, (L.rows() + 1) * sizeof(int));
  cudaMalloc(&d_col_ind, L.nonZeros() * sizeof(int));
  cudaMalloc(&d_values, L.nonZeros() * sizeof(double));

  cudaMemcpy(d_row_ptr, L.outerIndexPtr(), (L.rows() + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_col_ind, L.innerIndexPtr(), L.nonZeros() * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_values, L.valuePtr(), L.nonZeros() * sizeof(double),
             cudaMemcpyHostToDevice);

  // std::cout << "host u0[100 * nx + ny / 2]: " << u0.data()[100* nx + ny / 2]
  // << "\n"; std::cout << "host v0[100 * nx + ny / 2]: " << v0.data()[100* nx +
  // ny / 2] << "\n";
  auto gpu_start = std::chrono::high_resolution_clock::now();
  {
    Eigen::VectorX<double> u_past = (u0 - dt * v0);
    SGESolverDevice::Parameters params(num_snapshots, freq);
    SGESolverDevice solver(d_row_ptr, d_col_ind, d_values, c.data(), m.data(),
                           nx * ny, L.nonZeros(), u0.data(), v0.data(),
                           u_past.data(), params);
    for (uint32_t i = 1; i < nt; ++i) {
      solver.step(dt, i);
    }
    solver.transfer_snapshots(u_save_gpu.data(), 'u');
    solver.transfer_snapshots(v_save_gpu.data(), 'v');
  }
  auto gpu_end = std::chrono::high_resolution_clock::now();
  auto gpu_time =
      std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start)
          .count();

  Eigen::VectorXd u_diff = u_save_cpu - u_save_gpu;
  Eigen::VectorXd v_diff = v_save_cpu - v_save_gpu;

  std::cout << std::scientific << std::setprecision(4);
  std::cout << "u diff - L1: " << u_diff.lpNorm<1>()
            << ", L2: " << u_diff.norm() << "\n";
  std::cout << "v diff - L1: " << v_diff.lpNorm<1>()
            << ", L2: " << v_diff.norm() << "\n";
  std::cout << "host time:   " << cpu_time << " us\n";
  std::cout << "device time: " << gpu_time << " us\n";
  std::cout << "s: " << static_cast<double>(cpu_time) / gpu_time << "x\n";

  Eigen::Map<Eigen::Matrix<f_ty, -1, -1, Eigen::RowMajor>> u_cpu_mat(
      u_save_cpu.data(), num_snapshots, nx * ny);
  Eigen::Map<Eigen::Matrix<f_ty, -1, -1, Eigen::RowMajor>> u_gpu_mat(
      u_save_gpu.data(), num_snapshots, nx * ny);

  const std::vector<uint32_t> shape = {num_snapshots, nx, ny};
  const auto fname_u_cpu = "evolution_sg_u_host.npy";
  const auto fname_u_gpu = "evolution_sg_u_device.npy";

  save_to_npy(fname_u_cpu, u_save_cpu, shape);
  save_to_npy(fname_u_gpu, u_save_gpu, shape);

  cudaFree(d_row_ptr);
  cudaFree(d_col_ind);
  cudaFree(d_values);

  return 0;
}
