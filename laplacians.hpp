#include <Eigen/Sparse>

// TODO: finish both Sommerfeld radiation BCs and PBCs
// adapt from pytorch!

template <typename Float>
Eigen::SparseMatrix<Float> build_laplacian_noflux(uint32_t nx, uint32_t ny,
                                                  Float dx, Float dy) {
  // for this sparse solver we always assume nx == ny, dx == dy
  // which could of course be adapted in the future
  assert(nx == ny);
  assert(std::abs(dx - dy) < 1e-10);

  const uint32_t N = (nx + 2) * (nx + 2);
  const uint32_t nnz = N + 4 * (N - 1) - 4;
  using T = Eigen::Triplet<Float>;
  std::vector<T> triplets;
  triplets.reserve(nnz);

  for (uint32_t i = 0; i < N; ++i) {
    Float val = static_cast<Float>(-4.0);
    if (i < (nx + 2) || i >= N - (nx + 2) || i % (nx + 2) == 0 ||
        i % (nx + 2) == (nx + 1)) {
      val = static_cast<Float>(-3.0);
    }
    triplets.emplace_back(i, i, val);
  }

  for (uint32_t i = 0; i < N - 1; ++i) {
    if ((i + 1) % (nx + 2) != 0) {
      triplets.emplace_back(i, i + 1, static_cast<Float>(1.0));
      triplets.emplace_back(i + 1, i, static_cast<Float>(1.0));
    }
  }

  for (uint32_t i = 0; i < N - (nx + 2); ++i) {
    triplets.emplace_back(i, i + (nx + 2), static_cast<Float>(1.0));
    triplets.emplace_back(i + (nx + 2), i, static_cast<Float>(1.0));
  }

  // missing: corners!

  Eigen::SparseMatrix<Float> L(N, N);
  L.setFromTriplets(triplets.begin(), triplets.end());
  L.makeCompressed();
  L *= static_cast<Float>(1.0) / (dx * dy);

  return L;
}
