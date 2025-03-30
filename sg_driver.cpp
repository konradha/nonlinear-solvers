#include "sg_solver.hpp"
#include "util.hpp"

template <typename Float>
void neumann_bc_no_velocity(Eigen::VectorX<Float> & u,
                const uint32_t nx,
                const uint32_t ny) {
    // Used because we can save a lot of cycles that way.
    /*
     def neumann_bc(self, u, v, i, tau=None,):
        u[0, 1:-1] = u[1, 1:-1]
        u[-1, 1:-1] = u[-2, 1:-1]
        u[:, 0] = u[:, 1]
        u[:, -1] = u[:, -2]
     */
    Eigen::Map<Eigen::Matrix<Float, Eigen::Dynamic, Eigen::Dynamic>> u_map(u.data(), nx, ny);
    u_map.row(0).segment(1, ny-2) = u_map.row(1).segment(1, ny-2);
    u_map.row(nx-1).segment(1, ny-2) = u_map.row(nx-2).segment(1, ny-2);
    u_map.col(0) = u_map.col(1);
    u_map.col(ny-1) = u_map.col(ny-2);
}


template <typename Float, typename F>
Eigen::VectorX<Float> apply_function(const Eigen::VectorX<Float> &x,
                                     const Eigen::VectorX<Float> &y, F f) {
  const uint32_t nx = x.size();
  const uint32_t ny = y.size();
  Eigen::VectorX<Float> u(nx * ny);
  for (uint32_t i = 0; i < ny; ++i) {
    for (uint32_t j = 0; j < nx; ++j) {
      u[i * nx + j] = f(x[i], y[j]);
    }
  }
  return u;
}

template <typename Float, typename F>
Eigen::VectorX<Float> apply_function_uniform(Float x_min, Float x_max,
                                             uint32_t nx, Float y_min,
                                             Float y_max, uint32_t ny, F f) {
  Eigen::VectorX<Float> x = Eigen::VectorX<Float>::LinSpaced(nx, x_min, x_max);
  Eigen::VectorX<Float> y = Eigen::VectorX<Float>::LinSpaced(ny, y_min, y_max);
  return apply_function<Float>(x, y, f);
}

int main() {
  using f_ty = double;
  const uint32_t nx = 300, ny = 300;
  const f_ty Lx = 10., Ly = 10.;
  const f_ty dx = 2 * Lx / (nx + 1), dy = 2 * Ly / (ny + 1);
  const f_ty T = 5.;
  const uint32_t nt = 500;
  const uint32_t num_snapshots = 100;
  const auto freq = nt / num_snapshots;
  const auto dt = T / nt;

  // auto f = [](f_ty x, f_ty y) {
  //   return 2. * std::atan(std::exp(3. - 5. * std::sqrt(x * x + y * y)));
  // };

  auto f = [](f_ty x, f_ty y) {
    return 4. * std::atan(std::exp(3. - 5. * std::sqrt((x - y) * (x - y) / 3 + (x + y) * (x + y) / 2)));
  };
  auto zero = [](f_ty x, f_ty y) { return 0.; };
  auto one = [](f_ty x, f_ty y) { return 1.; };
  auto neg = [](f_ty x, f_ty y) { return -1.; };

  Eigen::VectorX<f_ty> u0 =
      apply_function_uniform<f_ty>(-Lx, Lx, nx, -Ly, Ly, ny, f);

  Eigen::VectorX<f_ty> v0 =
      apply_function_uniform<f_ty>(-Lx, Lx, nx, -Ly, Ly, ny, zero);

  Eigen::VectorX<f_ty> m =
      apply_function_uniform<f_ty>(-Lx, Lx, nx, -Ly, Ly, ny, one);

  Eigen::VectorX<f_ty> c =
      apply_function_uniform<f_ty>(-Lx, Lx, nx, -Ly, Ly, ny, one);

  Eigen::VectorX<f_ty> u_save(num_snapshots * nx * ny);
  Eigen::VectorX<f_ty> v_save(num_snapshots * nx * ny);

  Eigen::Map<Eigen::Matrix<f_ty, -1, -1, Eigen::RowMajor>>(
      u_save.data(), num_snapshots, nx * ny)
      .row(0) = (u0).transpose();
  Eigen::Map<Eigen::Matrix<f_ty, -1, -1, Eigen::RowMajor>>(
      v_save.data(), num_snapshots, nx * ny)
      .row(0) = (v0).transpose();

  const Eigen::SparseMatrix<f_ty> L =
      build_laplacian_noflux<f_ty>(nx - 2, ny - 2, dx, dy);
  Eigen::VectorX<f_ty> u = u0;
  Eigen::VectorX<f_ty> v = v0;
  Eigen::VectorX<f_ty> buf = v0;
  Eigen::VectorX<f_ty> u_past = u0 - dt * v0;
  Eigen::VectorX<f_ty> v_past = v0 - dt * dt * (L * u0);

  for (uint32_t i = 1; i < nt; ++i) {
    SGESolver::step_sv<f_ty>(u, u_past, buf, L, c, m, dt); 
    v = (u - u_past) / dt;
    neumann_bc_no_velocity<f_ty>(u, nx, ny);

    if (i % freq == 0) {
      Eigen::Map<Eigen::Matrix<f_ty, -1, -1, Eigen::RowMajor>>(
          u_save.data(), num_snapshots, nx * ny)
          .row(i / freq) = u.transpose();

      Eigen::Map<Eigen::Matrix<f_ty, -1, -1, Eigen::RowMajor>>(
          v_save.data(), num_snapshots, nx * ny)
          .row(i / freq) = v.transpose();
    }
    PROGRESS_BAR(i, nt);
  }

  const std::vector<uint32_t> shape = {num_snapshots, nx, ny};
  const auto fname_u = "evolution_sv_sg_u.npy";
  save_to_npy(fname_u, u_save, shape);
  const auto fname_v = "evolution_sv_sg_v.npy";
  save_to_npy(fname_v, v_save, shape);

  u_save = Eigen::VectorX<f_ty>::Zero(num_snapshots * nx * ny);
  v_save = Eigen::VectorX<f_ty>::Zero(num_snapshots * nx * ny);
  Eigen::Map<Eigen::Matrix<f_ty, -1, -1, Eigen::RowMajor>>(
      u_save.data(), num_snapshots, nx * ny)
      .row(0) = (u0).transpose();
  Eigen::Map<Eigen::Matrix<f_ty, -1, -1, Eigen::RowMajor>>(
      v_save.data(), num_snapshots, nx * ny)
      .row(0) = (v0).transpose();

  u = u0;
  v = v0;
  buf = v0;
  u_past = u0 - dt * v0;
  v_past = v0 - dt * dt * (L * u0);

  for (uint32_t i = 1; i < nt; i += 1) {   
    SGESolver::step<f_ty>(u, u_past, buf, L, c, m, dt); 
    v = (u - u_past) / (dt);
    neumann_bc_no_velocity<f_ty>(u, nx, ny);
    if (i % freq == 0) {
      Eigen::Map<Eigen::Matrix<f_ty, -1, -1, Eigen::RowMajor>>(
          u_save.data(), num_snapshots, nx * ny)
          .row(i / freq) = u.transpose();

      Eigen::Map<Eigen::Matrix<f_ty, -1, -1, Eigen::RowMajor>>(
          v_save.data(), num_snapshots, nx * ny)
          .row(i / freq) = v.transpose();
    }
    PROGRESS_BAR(i, nt);
  }
  const auto fname_ug = "evolution_gautschi_sg_u.npy";
  save_to_npy(fname_ug, u_save, shape);
  const auto fname_vg = "evolution_gautschi_sg_v.npy";
  save_to_npy(fname_vg, v_save, shape);
}
