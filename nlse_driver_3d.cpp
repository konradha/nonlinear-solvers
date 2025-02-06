#include "nlse_solver.hpp"
#include "util.hpp"

template <typename Float, typename F>
Eigen::VectorX<Float> apply_function(const Eigen::VectorX<Float> &x,
                                     const Eigen::VectorX<Float> &y,
                                     const Eigen::VectorX<Float> &z, F f) {
  const uint32_t nx = x.size();
  const uint32_t ny = y.size();
  const uint32_t nz = z.size();
  Eigen::VectorX<Float> u(nx * ny * nz);
  for (uint32_t k = 0; k < nz; ++k) {
    for (uint32_t j = 0; j < ny; ++j) {
      for (uint32_t i = 0; i < nx; ++i) {
        u[k * nx * ny + j * nx + i] = f(x[i], y[j], z[k]);
      }
    }
  }
  return u;
}

template <typename Float, typename F>
Eigen::VectorX<Float>
apply_function_uniform(Float x_min, Float x_max, uint32_t nx, Float y_min,
                       Float y_max, uint32_t ny, Float z_min, Float z_max,
                       uint32_t nz, F f) {
  Eigen::VectorX<Float> x = Eigen::VectorX<Float>::LinSpaced(nx, x_min, x_max);
  Eigen::VectorX<Float> y = Eigen::VectorX<Float>::LinSpaced(ny, y_min, y_max);
  Eigen::VectorX<Float> z = Eigen::VectorX<Float>::LinSpaced(nz, z_min, z_max);
  return apply_function<Float>(x, y, z, f);
}

template <typename Float>
void verify_laplacian_3d(uint32_t nx, uint32_t ny, uint32_t nz, Float L) {
  const Float dx = L / (nx - 1);
  const Float dy = L / (ny - 1);
  const Float dz = L / (nz - 1);

  const auto lap = build_laplacian_noflux_3d<Float>(nx, ny, nz, dx, dy, dz);

  auto index = [nx, ny](uint32_t i, uint32_t j, uint32_t k) {
    return k * (nx + 2) * (ny + 2) + j * (nx + 2) + i;
  };

  Eigen::VectorX<Float> u((nx + 2) * (ny + 2) * (nz + 2));
  Eigen::VectorX<Float> expected((nx + 2) * (ny + 2) * (nz + 2));

  u.setZero();
  expected.setZero();

  for (uint32_t k = 1; k < nz + 1; ++k) {
    for (uint32_t j = 1; j < ny + 1; ++j) {
      for (uint32_t i = 1; i < nx + 1; ++i) {
        Float x = -L / 2 + (i - 1) * dx;
        Float y = -L / 2 + (j - 1) * dy;
        Float z = -L / 2 + (k - 1) * dz;
        u[index(i, j, k)] = x * x + y * y + z * z;
        expected[index(i, j, k)] = 6.0;
      }
    }
  }

  Eigen::VectorX<Float> res = lap * u;

  Float max_diff = 0.0;
  Float l1_diff = 0.0;
  Float l2_diff = 0.0;
  uint32_t count = 0;

  for (uint32_t k = 1; k < nz + 1; ++k) {
    for (uint32_t j = 1; j < ny + 1; ++j) {
      for (uint32_t i = 1; i < nx + 1; ++i) {
        Float diff = std::abs(res[index(i, j, k)] - expected[index(i, j, k)]);
        max_diff = std::max(max_diff, diff);
        l1_diff += diff;
        l2_diff += diff * diff;
        count++;
      }
    }
  }

  l2_diff = std::sqrt(l2_diff);

  std::cout << "L1: " << l1_diff << "\n";
  std::cout << "L2: " << l2_diff << "\n";
  std::cout << "Max diff: " << max_diff << "\n";
}

int main() {
  using f_ty = double;
  using c_ty = std::complex<f_ty>;

  const uint32_t nx = 60, ny = 60, nz = 60;
  const f_ty Lx = 5., Ly = 5., Lz = 5.;
  const f_ty dx = 2 * Lx / (nx - 1), dy = 2 * Ly / (ny - 1),
             dz = 2 * Lz / (nz - 1);

  // verify_laplacian_3d<f_ty>(nx, ny, nz, Lx);

  const f_ty T = 1.5;
  const uint32_t nt = 500;
  const uint32_t num_snapshots = 100;
  const auto freq = nt / num_snapshots;

  const auto dt = T / nt;
  c_ty dti = c_ty(0, dt);

  f_ty R = 3.0;
  f_ty a = 0.5;
  f_ty omega = 0.5;

  auto f_vortex_ring = [&](c_ty x, c_ty y, c_ty z) {
    f_ty r = std::sqrt(x.real() * x.real() + y.real() * y.real());
    f_ty phi = std::atan2(y.real(), x.real());
    f_ty rho = std::sqrt((r - R) * (r - R) + z.real() * z.real());
    f_ty theta = std::atan2(z.real(), r - R);

    c_ty phase = c_ty(0, 1) * phi;
    f_ty amplitude = std::tanh(rho / a);

    return amplitude * std::exp(phase) * std::exp(-rho * rho / (4.0 * a * a));
  };

  auto gaussian = [&](c_ty x, c_ty y, c_ty z) {
    const f_ty sigma = 0.5;
    const f_ty x0 = 2.0, y0 = 2.0, z0 = 2.0;
    return std::exp(
        -((x - x0) * (x - x0) + (y - y0) * (y - y0) + (z - z0) * (z - z0)) /
        (2 * sigma * sigma));
  };

  auto wavepacket = [&](c_ty x, c_ty y, c_ty z) {
    const f_ty sigma = 0.5;
    const f_ty kx = 6.0;
    const f_ty norm = 1.0 / std::pow(2.0 * M_PI * sigma * sigma, 3.0 / 4.0);
    f_ty wave_fac = std::real(x) > 0 ? std::real(x) : -std::real(x);
    std::complex<f_ty> phase(0.0, kx * wave_fac);
    return norm * std::exp(-(x * x + y * y + z * z) / (2.0 * sigma * sigma)) *
           std::exp(phase);
  };

  auto vortex_pair = [&](c_ty x, c_ty y, c_ty z) {
    const f_ty d = 0.5;
    const f_ty sigma = 0.2;
    const int m1 = 1;
    const int m2 = -1;

    f_ty y_real = std::real(y);
    f_ty x_real = std::real(x);
    f_ty r1 = std::sqrt((x_real + d) * (x_real + d) + y_real * y_real);
    f_ty r2 = std::sqrt((x_real - d) * (x_real - d) + y_real * y_real);
    f_ty theta1 = std::atan2(std::real(y), std::real(x + d));
    f_ty theta2 = std::atan2(std::real(y), std::real(x - d));

    std::complex<f_ty> phase1(0.0, m1 * theta1);
    std::complex<f_ty> phase2(0.0, m2 * theta2);

    return std::exp(-(r1 * r1) / (2.0 * sigma * sigma)) * std::exp(phase1) +
           std::exp(-(r2 * r2) / (2.0 * sigma * sigma)) * std::exp(phase2) *
               std::exp(-(z * z) / (2.0 * sigma * sigma));
  };

  auto spatial_soliton = [&](c_ty x, c_ty y, c_ty z) {
    const f_ty sigma = 0.4;
    const f_ty kx = 4.0;
    const f_ty ky = 3.0;
    const f_ty kz = 2.0;
    const f_ty A = 2.0;

    const f_ty norm = 1.0 / std::pow(2.0 * M_PI * sigma * sigma, 3.0 / 4.0);

    std::complex<f_ty> phase(0.0, kx * std::real(x) + ky * std::real(y) +
                                      kz * std::real(z));
    return A * norm *
           std::exp(-(x * x + y * y + z * z) / (2.0 * sigma * sigma)) *
           std::exp(phase) *
           (1.0 + 0.2 * std::cos(2.0 * kx * std::real(x)) *
                      std::cos(2.0 * ky * std::real(y)) *
                      std::cos(2.0 * kz * std::real(z)));
  };

  Eigen::VectorX<c_ty> u0 = apply_function_uniform<c_ty>(
      -Lx, Lx, nx, -Ly, Ly, ny, -Lz, Lz, nz, f_vortex_ring);

  auto get_norm = [&](const Eigen::VectorX<c_ty> &x) {
    return std::sqrt((x.array().abs2() * dx * dy * dz).sum());
  };
  const auto Norm = get_norm(u0);
  u0 = u0 / Norm;

  Eigen::VectorX<c_ty> u_save(num_snapshots * nx * ny * nz);
  Eigen::Map<Eigen::Matrix<c_ty, -1, -1, Eigen::RowMajor>>(
      u_save.data(), num_snapshots, nx * ny * nz)
      .row(0) = u0.transpose();

  Eigen::VectorX<c_ty> u = u0;
  Eigen::VectorX<c_ty> buf = u0;
  Eigen::VectorX<c_ty> rho_buf = u0;

  const Eigen::SparseMatrix<c_ty> L =
      build_laplacian_noflux_3d<c_ty>(nx - 2, ny - 2, nz - 2, dx, dy, dz);

  for (uint32_t i = 1; i < nt; ++i) {
    NLSESolver::step<c_ty>(buf, rho_buf, u, L, dti);

    if (i % freq == 0) {
      Eigen::Map<Eigen::Matrix<c_ty, -1, -1, Eigen::RowMajor>>(
          u_save.data(), num_snapshots, nx * ny * nz)
          .row(i / freq) = u.transpose();
    }
    PROGRESS_BAR(i, nt);
  }

  const std::vector<uint32_t> shape = {num_snapshots, nx, ny, nz};
  const auto fname_u = "evolution_nlse_3d.npy";
  save_to_npy(fname_u, u_save, shape);
}
