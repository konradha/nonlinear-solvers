#include "nlse_solver.hpp"
#include "util.hpp"

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
  using c_ty = std::complex<f_ty>;

  const uint32_t nx = 100, ny = 100;
  const f_ty Lx = 10., Ly = 10.;
  const f_ty dx = 2 * Lx / (nx - 1), dy = 2 * Ly / (ny - 1);

  const f_ty T = 2.5;
  const uint32_t nt = 100;
  const uint32_t num_snapshots = 100;
  
  const auto dt = T / nt;
  c_ty dti = c_ty(0, dt);

  auto f = [&](c_ty x, c_ty y) { return std::exp(-(x * x + y * y)); };

  f_ty k0 = .5;
  f_ty theta = 3.14 * .25;
  f_ty kx = k0 * std::cos(theta);
  f_ty ky = k0 * std::sin(theta);
  f_ty sigma = .3;
  f_ty A = 1.;

  auto f_collision = [&](c_ty x, c_ty y) {
      const auto un1 = A * std::exp(-((x - 3.) * (x - 3.) + (y - 3.) * (y - 3.)) / 4. / sigma / sigma) *
          std::exp(c_ty(0, -1.) * (x + y));
      const auto un2 = A * std::exp(-((x + 3.) * (x + 3.) + (y + 3.) * (y + 3.)) / 4. / sigma / sigma) *
          std::exp(c_ty(0, 1.) * (x + y));
      const auto un = un1 + un2;
      return un; 
  };

  Eigen::VectorX<c_ty> u0 =
      apply_function_uniform<c_ty>(-Lx, Lx, nx, -Ly, Ly, ny, f_collision);

  u0 = u0 / u0.norm();

  Eigen::VectorX<f_ty> u_save(num_snapshots * nx * ny);
  Eigen::Map<Eigen::Matrix<f_ty, -1, -1, Eigen::RowMajor>>(
      u_save.data(), num_snapshots, nx * ny)
      .row(0) = (
              u0.real().cwiseProduct(u0.real()) + u0.imag().cwiseProduct(u0.imag())
              ).transpose(); 

  Eigen::VectorX<c_ty> u = u0;
  Eigen::VectorX<c_ty> buf = u0;
  Eigen::VectorX<c_ty> rho_buf = u0;

  const Eigen::SparseMatrix<c_ty> L = build_laplacian_noflux<c_ty>(nx-2, ny-2, dx, dy); 
  for(uint32_t i=1; i < nt; ++i){ 
    NLSESolver::step<c_ty>(buf, rho_buf, u, L, dti);

    Eigen::Map<Eigen::Matrix<f_ty, -1, -1, Eigen::RowMajor>>(
    u_save.data(), num_snapshots, nx * ny)
    .row(i) = (u.real().cwiseProduct(u.real()) + u.imag().cwiseProduct(u.imag())
              ).transpose();
    PROGRESS_BAR(i, nt);
  }

  const std::vector<uint32_t> shape = {num_snapshots, nx, ny}; 
  const auto fname_u = "evolution.npy";
  save_to_npy(fname_u, u_save, shape);
}
