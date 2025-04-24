#ifndef BOUNDARIES_HPP
#define BOUNDARIES_HPP

#include <Eigen/Dense>

// We assume all vectors u, v to be actual vectors, ie. their shapes to be
// either (nx * ny, 1) or (nx * ny * nz, 1)

template <typename Float>
void neumann_bc(Eigen::VectorX<Float> &u, Eigen::VectorX<Float> &v,
                const uint32_t nx, const uint32_t ny, const Float dx,
                const Float dy) {
  // port of the Numpy-y
  /*
   def neumann_bc(self, u, v, i, tau=None,):
      u[0, 1:-1] = u[1, 1:-1]
      u[-1, 1:-1] = u[-2, 1:-1]
      u[:, 0] = u[:, 1]
      u[:, -1] = u[:, -2]

      v[0, 1:-1] = 0
      v[-1, 1:-1] = 0
      v[1:-1, 0] = 0
      v[1:-1, -1] = 0
   */
  Eigen::Map<Eigen::Matrix<Float, Eigen::Dynamic, Eigen::Dynamic>> u_map(
      u.data(), nx, ny);
  Eigen::Map<Eigen::Matrix<Float, Eigen::Dynamic, Eigen::Dynamic>> v_map(
      v.data(), nx, ny);
  u_map.row(0).segment(1, ny - 2) = u_map.row(1).segment(1, ny - 2);
  u_map.row(nx - 1).segment(1, ny - 2) = u_map.row(nx - 2).segment(1, ny - 2);
  u_map.col(0) = u_map.col(1);
  u_map.col(ny - 1) = u_map.col(ny - 2);
  v_map.row(0).segment(1, ny - 2).setZero();
  v_map.row(nx - 1).segment(1, ny - 2).setZero();
  v_map.block(1, 0, nx - 2, 1).setZero();
  v_map.block(1, ny - 1, nx - 2, 1).setZero();
}

template <typename Float>
void neumann_bc_no_velocity(Eigen::VectorX<Float> &u, const uint32_t nx,
                            const uint32_t ny) {
  // Used because we can save a lot of cycles that way.
  /*
   def neumann_bc(self, u, v, i, tau=None,):
      u[0, 1:-1] = u[1, 1:-1]
      u[-1, 1:-1] = u[-2, 1:-1]
      u[:, 0] = u[:, 1]
      u[:, -1] = u[:, -2]
   */
  Eigen::Map<Eigen::Matrix<Float, Eigen::Dynamic, Eigen::Dynamic>> u_map(
      u.data(), nx, ny);
  u_map.row(0).segment(1, ny - 2) = u_map.row(1).segment(1, ny - 2);
  u_map.row(nx - 1).segment(1, ny - 2) = u_map.row(nx - 2).segment(1, ny - 2);
  u_map.col(0) = u_map.col(1);
  u_map.col(ny - 1) = u_map.col(ny - 2);
}

void experimental_nlse_envelope_bc(
    Eigen::VectorX<std::complex<double>> &u, uint32_t nx, uint32_t ny,
    double dx, double dy, const Eigen::VectorX<std::complex<double>> &m) {

  Eigen::VectorX<std::complex<double>> u_copy = u;

  auto idx = [nx, ny](uint32_t i, uint32_t j) { return i * (ny + 2) + j; };

  auto compute_effective_k = [&](uint32_t i, uint32_t j, double h) {
    std::complex<double> laplacian =
        (u_copy[idx(i + 1, j)] + u_copy[idx(i - 1, j)] + u_copy[idx(i, j + 1)] +
         u_copy[idx(i, j - 1)] - 4.0 * u_copy[idx(i, j)]) /
        (h * h);

    std::complex<double> nonlinear_term =
        m[idx(i, j)] * std::norm(u_copy[idx(i, j)]);
    std::complex<double> k_squared =
        -laplacian / u_copy[idx(i, j)] + nonlinear_term;

    double k_squared_real = std::real(k_squared);

    if (!std::isfinite(k_squared_real) || k_squared_real < 0) {
      k_squared_real = std::abs(nonlinear_term);
    }

    double nyquist_limit = 2.0 / (h * h);

    if (k_squared_real > nyquist_limit) {
      k_squared_real = nyquist_limit;
    }

    return std::sqrt(k_squared_real);
  };

  for (uint32_t j = 1; j <= ny; ++j) {
    double k_left = compute_effective_k(1, j, dx);
    std::complex<double> phase_left =
        std::exp(std::complex<double>(0, -k_left * dx));
    u[idx(0, j)] = phase_left * u_copy[idx(1, j)];

    double k_right = compute_effective_k(nx, j, dx);
    std::complex<double> phase_right =
        std::exp(std::complex<double>(0, -k_right * dx));
    u[idx(nx + 1, j)] = phase_right * u_copy[idx(nx, j)];
  }

  for (uint32_t i = 1; i <= nx; ++i) {
    double k_bottom = compute_effective_k(i, 1, dy);
    std::complex<double> phase_bottom =
        std::exp(std::complex<double>(0, -k_bottom * dy));
    u[idx(i, 0)] = phase_bottom * u_copy[idx(i, 1)];

    double k_top = compute_effective_k(i, ny, dy);
    std::complex<double> phase_top =
        std::exp(std::complex<double>(0, -k_top * dy));
    u[idx(i, ny + 1)] = phase_top * u_copy[idx(i, ny)];
  }

  u[idx(0, 0)] = 0.5 * (u[idx(0, 1)] + u[idx(1, 0)]);
  u[idx(0, ny + 1)] = 0.5 * (u[idx(0, ny)] + u[idx(1, ny + 1)]);
  u[idx(nx + 1, 0)] = 0.5 * (u[idx(nx, 0)] + u[idx(nx + 1, 1)]);
  u[idx(nx + 1, ny + 1)] = 0.5 * (u[idx(nx, ny + 1)] + u[idx(nx + 1, ny)]);
}

#endif
