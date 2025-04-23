#include "boundaries.hpp"
#include "cmdline_parser.hpp"
#include "eigen_krylov_complex.hpp"
#include "laplacians.hpp"
#include "nlse_cubic_solver.hpp"
#include "nlse_cubic_solver_3d.hpp"
#include "nlse_cubic_gautschi_solver.hpp"
#include "nlse_cubic_gautschi_solver_3d.hpp"
#include "nlse_cubic_quintic_solver.hpp"
#include "nlse_cubic_quintic_gautschi_solver.hpp"
#include "nlse_saturating_solver.hpp"
#include "nlse_saturating_gautschi_solver.hpp"
#include "util.hpp"

#ifdef HAVE_CUDA
#include "boundaries.cuh"
#include "nlse_dev.hpp"
#include "nlse_cubic_quintic_dev.hpp"
#include "nlse_saturating_dev.hpp"
#endif

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <chrono>
#include <complex>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

template <typename Scalar_t>
Eigen::VectorX<Scalar_t> load_or_default(const std::optional<std::string>& file_path, 
                                         uint32_t nx, uint32_t ny, Scalar_t default_value) {
    if (file_path) {
        try {
            std::vector<uint32_t> shape;
            Eigen::VectorX<Scalar_t> data = read_from_npy<Scalar_t>(file_path.value(), shape);
            
            if (shape.size() != 2 || shape[0] != ny || shape[1] != nx) {
                std::cerr << "Error: Array dimensions mismatch\n";
                std::cerr << "Expected: " << ny << "x" << nx << "\n";
                std::cerr << "Got: " << shape[0] << "x" << shape[1] << "\n";
                std::cerr << "Using default value " << default_value << " everywhere\n";
                return Eigen::VectorX<Scalar_t>::Constant(nx * ny, default_value);
            }
            return data;
        } catch (const std::exception& e) {
            std::cerr << "Error loading file " << file_path.value() << ": " << e.what() << "\n";
            std::cerr << "Using default value " << default_value << " everywhere\n";
            return Eigen::VectorX<Scalar_t>::Constant(nx * ny, default_value);
        }
    }
    return Eigen::VectorX<Scalar_t>::Constant(nx * ny, default_value);
}

template <typename Scalar_t>
Eigen::VectorX<Scalar_t> load_or_default_3d(const std::optional<std::string>& file_path, 
                                           uint32_t nx, uint32_t ny, uint32_t nz, Scalar_t default_value) {
    if (file_path) {
        try {
            std::vector<uint32_t> shape;
            Eigen::VectorX<Scalar_t> data = read_from_npy<Scalar_t>(file_path.value(), shape);
            
            if (shape.size() != 3 || shape[0] != nz || shape[1] != ny || shape[2] != nx) {
                std::cerr << "Error: Array dimensions mismatch\n";
                std::cerr << "Expected: " << nz << "x" << ny << "x" << nx << "\n";
                std::cerr << "Got: " << shape[0] << "x" << shape[1] << "x" << shape[2] << "\n";
                std::cerr << "Using default value " << default_value << " everywhere\n";
                return Eigen::VectorX<Scalar_t>::Constant(nx * ny * nz, default_value);
            }
            return data;
        } catch (const std::exception& e) {
            std::cerr << "Error loading file " << file_path.value() << ": " << e.what() << "\n";
            std::cerr << "Using default value " << default_value << " everywhere\n";
            return Eigen::VectorX<Scalar_t>::Constant(nx * ny * nz, default_value);
        }
    }
    return Eigen::VectorX<Scalar_t>::Constant(nx * ny * nz, default_value);
}

void run_host_2d(const CommandLineArgs& args) {
    const uint32_t nx = args.n;
    const uint32_t ny = args.n;
    const double dx = 2 * args.L / (nx - 1);
    const double dy = 2 * args.L / (ny - 1);
    const double dt = args.T / args.nt;
    const std::complex<double> dti(0, dt);
    const auto freq = args.nt / args.snapshots;
    
    std::vector<uint32_t> input_shape;
    Eigen::VectorXcd u0 = read_from_npy<std::complex<double>>(args.initial_u_file, input_shape);
    
    if (input_shape.size() != 2 || input_shape[0] != ny || input_shape[1] != nx) {
        std::cerr << "Error: Input array dimensions mismatch\n";
        std::cerr << "Expected: " << ny << "x" << nx << "\n";
        std::cerr << "Got: " << input_shape[0] << "x" << input_shape[1] << "\n";
        exit(1);
    }
    
    Eigen::VectorXd m = load_or_default<double>(args.focussing_file, nx, ny, 1.0);
    Eigen::VectorXd c = load_or_default<double>(args.anisotropy_file, nx, ny, 1.0);
    
    Eigen::SparseMatrix<std::complex<double>> L;
    if (args.anisotropy_file) {
        L = build_anisotropic_laplacian_noflux<std::complex<double>>(nx - 2, ny - 2, dx, dy, c);
    } else {
        L = build_laplacian_noflux<std::complex<double>>(nx - 2, ny - 2, dx, dy);
    }
    
    Eigen::VectorXcd u_save(args.snapshots * nx * ny);
    Eigen::Map<Eigen::Matrix<std::complex<double>, -1, -1, Eigen::RowMajor>> u_save_mat(
        u_save.data(), args.snapshots, nx * ny);
    
    u_save_mat.row(0) = u0.transpose();
    
    Eigen::VectorXcd u = u0;
    Eigen::VectorXcd buf = u0;
    Eigen::VectorXcd rho_buf(nx * ny);
    
    for (uint32_t i = 1; i < args.nt; ++i) {
        switch (args.system_type) {
            case SystemType::NLSE_CUBIC:
                if (args.method == IntegrationMethod::GAUTSCHI) {
                    NLSECubicGautschiSolver::step<std::complex<double>>(u, buf, rho_buf, L, m, dti);
                } else {
                    NLSESolver::step<std::complex<double>>(buf, rho_buf, u, L, m, dti);
                }
                break;
            case SystemType::NLSE_CUBIC_QUINTIC:
                if (args.method == IntegrationMethod::GAUTSCHI) {
                    NLSECubicQuinticGautschiSolver::step<std::complex<double>>(u, buf, rho_buf, L, m, dti);
                } else {
                    NLSECubicQuinticSolver::step<std::complex<double>>(buf, rho_buf, u, L, m, dti);
                }
                break;
            case SystemType::NLSE_SATURABLE:
                if (args.method == IntegrationMethod::GAUTSCHI) {
                    NLSESaturatingGautschiSolver::step<std::complex<double>>(u, buf, rho_buf, L, m, dti);
                } else {
                    NLSESaturatingSolver::step<std::complex<double>>(buf, rho_buf, u, L, m, dti);
                }
                break;
            default:
                std::cerr << "Unsupported system type for NLSE\n";
                exit(1);
        }
        
        neumann_bc_no_velocity<std::complex<double>>(u, nx, ny);
        
        if (i % freq == 0) {
            uint32_t snapshot_idx = i / freq;
            if (snapshot_idx < args.snapshots) {
                u_save_mat.row(snapshot_idx) = u.transpose();
            }
        }
    }
    
    const std::vector<uint32_t> shape = {args.snapshots, ny, nx};
    save_to_npy(args.trajectory_file, u_save, shape);
}

void run_host_3d(const CommandLineArgs& args) {
    const uint32_t nx = args.n;
    const uint32_t ny = args.n;
    const uint32_t nz = args.n;
    const double dx = 2 * args.L / (nx - 1);
    const double dy = 2 * args.L / (ny - 1);
    const double dz = 2 * args.L / (nz - 1);
    const double dt = args.T / args.nt;
    const std::complex<double> dti(0, dt);
    const auto freq = args.nt / args.snapshots;
    
    std::vector<uint32_t> input_shape;
    Eigen::VectorXcd u0 = read_from_npy<std::complex<double>>(args.initial_u_file, input_shape);
    
    if (input_shape.size() != 3 || input_shape[0] != nz || input_shape[1] != ny || input_shape[2] != nx) {
        std::cerr << "Error: Input array dimensions mismatch\n";
        std::cerr << "Expected: " << nz << "x" << ny << "x" << nx << "\n";
        std::cerr << "Got: " << input_shape[0] << "x" << input_shape[1] << "x" << input_shape[2] << "\n";
        exit(1);
    }
    
    Eigen::VectorXd m = load_or_default_3d<double>(args.focussing_file, nx, ny, nz, 1.0);
    Eigen::VectorXd c = load_or_default_3d<double>(args.anisotropy_file, nx, ny, nz, 1.0);
    
    Eigen::SparseMatrix<std::complex<double>> L;
    if (args.anisotropy_file) {
        L = build_anisotropic_laplacian_noflux_3d<std::complex<double>>(nx - 2, ny - 2, nz - 2, dx, dy, dz, c);
    } else {
        L = build_laplacian_noflux_3d<std::complex<double>>(nx - 2, ny - 2, nz - 2, dx, dy, dz);
    }
    
    Eigen::VectorXcd u_save(args.snapshots * nx * ny * nz);
    Eigen::Map<Eigen::Matrix<std::complex<double>, -1, -1, Eigen::RowMajor>> u_save_mat(
        u_save.data(), args.snapshots, nx * ny * nz);
    
    u_save_mat.row(0) = u0.transpose();
    
    Eigen::VectorXcd u = u0;
    Eigen::VectorXcd buf = u0;
    Eigen::VectorXcd rho_buf(nx * ny * nz);
    
    for (uint32_t i = 1; i < args.nt; ++i) {
        switch (args.system_type) {
            case SystemType::NLSE_CUBIC:
                if (args.method == IntegrationMethod::GAUTSCHI) {
                    NLSECubicGautschiSolver3D::step<std::complex<double>>(u, buf, rho_buf, L, m, dti);
                } else {
                    NLSESolver3D::step<std::complex<double>>(buf, rho_buf, u, L, m, dti);
                }
                break;
            default:
                std::cerr << "Unsupported 3D system type for NLSE\n";
                exit(1);
        }
        
        neumann_bc_no_velocity_3d<std::complex<double>>(u, nx, ny, nz);
        
        if (i % freq == 0) {
            uint32_t snapshot_idx = i / freq;
            if (snapshot_idx < args.snapshots) {
                u_save_mat.row(snapshot_idx) = u.transpose();
            }
        }
    }
    
    const std::vector<uint32_t> shape = {args.snapshots, nz, ny, nx};
    save_to_npy(args.trajectory_file, u_save, shape);
}

#ifdef HAVE_CUDA
void run_device_2d(const CommandLineArgs& args) {
    const uint32_t nx = args.n;
    const uint32_t ny = args.n;
    const double dx = 2 * args.L / (nx - 1);
    const double dy = 2 * args.L / (ny - 1);
    const double dt = args.T / args.nt;
    const std::complex<double> dti(0, dt);
    const auto freq = args.nt / args.snapshots;
    
    std::vector<uint32_t> input_shape;
    Eigen::VectorXcd u0 = read_from_npy<std::complex<double>>(args.initial_u_file, input_shape);
    
    if (input_shape.size() != 2 || input_shape[0] != ny || input_shape[1] != nx) {
        std::cerr << "Error: Input array dimensions mismatch\n";
        std::cerr << "Expected: " << ny << "x" << nx << "\n";
        std::cerr << "Got: " << input_shape[0] << "x" << input_shape[1] << "\n";
        exit(1);
    }
    
    Eigen::VectorXd m = load_or_default<double>(args.focussing_file, nx, ny, 1.0);
    Eigen::VectorXd c = load_or_default<double>(args.anisotropy_file, nx, ny, 1.0);
    
    Eigen::SparseMatrix<std::complex<double>> L;
    if (args.anisotropy_file) {
        L = build_anisotropic_laplacian_noflux<std::complex<double>>(nx - 2, ny - 2, dx, dy, c);
    } else {
        L = build_laplacian_noflux<std::complex<double>>(nx - 2, ny - 2, dx, dy);
    }
    
    Eigen::VectorXcd u_save(args.snapshots * nx * ny);
    
    device::NLSESolverDevice::Parameters params(args.snapshots, freq, 15);
    
    switch (args.system_type) {
        case SystemType::NLSE_CUBIC: {
            device::NLSESolverDevice solver(L, u0.data(), m.data(), params);
            
            for (uint32_t i = 1; i < args.nt; ++i) {
                solver.step(dti, i);
                solver.apply_bc();
            }
            
            solver.transfer_snapshots(u_save.data());
            break;
        }
        case SystemType::NLSE_CUBIC_QUINTIC: {
            device::NLSECubicQuinticSolverDevice solver(L, u0.data(), m.data(), params);
            
            for (uint32_t i = 1; i < args.nt; ++i) {
                solver.step(dti, i);
                solver.apply_bc();
            }
            
            solver.transfer_snapshots(u_save.data());
            break;
        }
        case SystemType::NLSE_SATURABLE: {
            device::NLSESaturatingSolverDevice solver(L, u0.data(), m.data(), params);
            
            for (uint32_t i = 1; i < args.nt; ++i) {
                solver.step(dti, i);
                solver.apply_bc();
            }
            
            solver.transfer_snapshots(u_save.data());
            break;
        }
        default:
            std::cerr << "Unsupported system type for NLSE on device\n";
            exit(1);
    }
    
    const std::vector<uint32_t> shape = {args.snapshots, ny, nx};
    save_to_npy(args.trajectory_file, u_save, shape);
}
#endif

int main(int argc, char **argv) {
    CommandLineArgs args = CommandLineParser::parse_nlse_args(argc, argv);
    
    if (args.device == DeviceType::HOST) {
        if (args.dim == 2) {
            run_host_2d(args);
        } else if (args.dim == 3) {
            run_host_3d(args);
        }
    } else {
#ifdef HAVE_CUDA
        if (args.dim == 2) {
            run_device_2d(args);
        } else {
            std::cerr << "3D simulations on device not yet implemented\n";
            exit(1);
        }
#else
        std::cerr << "CUDA support not enabled. Recompile with -DENABLE_GPU=ON\n";
        exit(1);
#endif
    }
    
    return 0;
}
