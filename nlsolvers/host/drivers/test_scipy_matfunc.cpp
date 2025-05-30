#include "eigen_krylov_complex.hpp"
#include "eigen_krylov_real.hpp"

#include "laplacians.hpp"
#include "util.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>


int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " n L outfile_base" << std::endl;
    return 1;
  }

  GridInfo grid;
  const double t = 1e-2;
  std::string outfile_base;
  int n_side = 0;


  try {
    n_side = std::stoi(argv[1]);
    grid.Lx = std::stod(argv[2]);
    outfile_base = argv[3];
    if (n_side <= 0 || grid.Lx <= 0.0 || t <= 0.0)
      throw std::invalid_argument("Invalid parameters");
  } catch (const std::exception &e) {
    std::cerr << "Error parsing arguments: " << e.what() << std::endl;
    return 1;
  }

  grid.is_3d = true;
  grid.nx = grid.ny = grid.nz = n_side;
  grid.Ly = grid.Lz = grid.Lx;
  grid.total_size = static_cast<long>(grid.nx) * grid.ny * grid.nz;
  grid.shape[0] = grid.nx;
  grid.shape[1] = grid.ny;
  grid.shape[2] = grid.nz;
  
  grid.dx = 2.0 * grid.Lx / grid.nx;
  grid.dy = 2.0 * grid.Ly / grid.ny;
  grid.dz = 2.0 * grid.Lz / grid.nz;


  try {
    Eigen::VectorXd c_coeff = Eigen::VectorXd::Ones(grid.total_size);
    std::cout << " Try-block, after c(x,y,z)\n";
    Eigen::SparseMatrix<double> L_op_real =
        (build_anisotropic_laplacian_noflux_3d<double>(
             grid.nx-2, grid.ny-2, grid.nz-2, grid.dx, grid.dy, grid.dz, c_coeff))
            .eval();
    Eigen::SparseMatrix<std::complex<double>> L_op_complex =
        (build_anisotropic_laplacian_noflux_3d<std::complex<double>>(
             grid.nx-2, grid.ny-2, grid.nz-2, grid.dx, grid.dy, grid.dz, c_coeff))
            .eval();

    std::cout << " Try-block, after matrices\n";
       
    if (L_op_real.rows() != grid.total_size)
      throw std::runtime_error("Laplacian size mismatch");

    double gaussian_width = grid.Lx / 5.0;
    Eigen::VectorXd u0_real =
        create_centered_gaussian_3d<double>(grid, gaussian_width);
    Eigen::VectorXcd u0_complex = u0_real.cast<std::complex<double>>();

    std::vector<int> krylov_dims = {10, 20, 30, 40, 50};
    std::complex<double> complex_dt(0.0, t);

    save_sparse_matrix_csr_components_npy(outfile_base + "_L_real", L_op_real);
    save_sparse_matrix_csr_components_npy(outfile_base + "_L_complex",
                                          L_op_complex);

    save_to_npy(outfile_base + "_u0_real.npy",  u0_real, {(uint32_t)grid.total_size});
    save_to_npy(outfile_base + "_u0_complex.npy",  u0_complex, {(uint32_t)grid.total_size});
    Eigen::VectorX<uint32_t> dims(3); dims(0) = grid.nx; dims(1) = grid.nx; dims(2) = grid.nx;
    save_to_npy(outfile_base + "_dims.npy",  dims, {(uint32_t)(3)}); 

    std::cout << "Instantiated all data, computing now\n";
   

    for (int m : krylov_dims) {
      Eigen::VectorXd y_host_real =
          expm_multiply(L_op_real, u0_real, t, m);
      Eigen::VectorXcd y_host_complex =
          expm_multiply(L_op_complex, u0_complex, complex_dt, m);

      save_to_npy(+ "_y_host_real_m" + std::to_string(m) +
                          ".npy",  y_host_real, {(uint32_t)grid.total_size});
      save_to_npy(+ "_y_host_complex_m" + std::to_string(m) +
                          ".npy",  y_host_complex, {(uint32_t)grid.total_size});
    }

  } catch (const std::exception &e) {
    std::cerr << "Error during computation or saving: " << e.what()
              << std::endl;
    return 1;
  }

  return 0;
}
