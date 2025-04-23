#include "boundaries.hpp"
#include "cmdline_parser.hpp"
#include "eigen_krylov_real.hpp"
#include "laplacians.hpp"
#include "kg_solver.hpp"
#include "kg_solver_3d.hpp"
#include "kg_sv_solver.hpp"
#include "kg_sv_solver_3d.hpp"
#include "sg_single_solver.hpp"
#include "sg_single_sv_solver.hpp"
#include "sg_double_solver.hpp"
#include "sg_double_sv_solver.hpp"
#include "sg_hyperbolic_solver.hpp"
#include "sg_hyperbolic_sv_solver.hpp"
#include "phi4_solver.hpp"
#include "phi4_sv_solver.hpp"
#include "util.hpp"

#ifdef HAVE_CUDA
#include "boundaries.cuh"
#include "kg_dev.hpp"
#include "sg_single_dev.hpp"
#include "sg_double_dev.hpp"
#include "sg_hyperbolic_dev.hpp"
#include "phi4_dev.hpp"
#endif

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <chrono>
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
    const auto freq = args.nt / args.snapshots;
    
    std::vector<uint32_t> input_shape;
    Eigen::VectorXd u0 = read_from_npy<double>(args.initial_u_file, input_shape);
    
    if (input_shape.size() != 2 || input_shape[0] != ny || input_shape[1] != nx) {
        std::cerr << "Error: Input array dimensions mismatch\n";
        std::cerr << "Expected: " << ny << "x" << nx << "\n";
        std::cerr << "Got: " << input_shape[0] << "x" << input_shape[1] << "\n";
        exit(1);
    }
    
    Eigen::VectorXd v0;
    if (args.initial_v_file) {
        v0 = read_from_npy<double>(args.initial_v_file.value(), input_shape);
        if (input_shape.size() != 2 || input_shape[0] != ny || input_shape[1] != nx) {
            std::cerr << "Error: Velocity array dimensions mismatch\n";
            std::cerr << "Expected: " << ny << "x" << nx << "\n";
            std::cerr << "Got: " << input_shape[0] << "x" << input_shape[1] << "\n";
            exit(1);
        }
    } else {
        v0 = Eigen::VectorXd::Zero(nx * ny);
    }
    
    Eigen::VectorXd u_past = u0 - dt * v0;
    
    Eigen::VectorXd m = load_or_default<double>(args.focussing_file, nx, ny, 1.0);
    Eigen::VectorXd c = load_or_default<double>(args.anisotropy_file, nx, ny, 1.0);
    
    Eigen::SparseMatrix<double> L;
    if (args.anisotropy_file) {
        L = (-(build_anisotropic_laplacian_noflux<double>(nx - 2, ny - 2, dx, dy, c))).eval();
    } else {
        L = (-(build_laplacian_noflux<double>(nx - 2, ny - 2, dx, dy))).eval();
    }
    
    Eigen::VectorXd u_save(args.snapshots * nx * ny);
    Eigen::VectorXd v_save;
    
    if (args.velocity_file) {
        v_save.resize(args.snapshots * nx * ny);
    }
    
    Eigen::Map<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>> u_save_mat(
        u_save.data(), args.snapshots, nx * ny);
    
    Eigen::Map<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>> v_save_mat;
    if (args.velocity_file) {
        new (&v_save_mat) Eigen::Map<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(
            v_save.data(), args.snapshots, nx * ny);
    }
    
    u_save_mat.row(0) = u0.transpose();
    if (args.velocity_file) {
        v_save_mat.row(0) = v0.transpose();
    }
    
    Eigen::VectorXd u = u0;
    Eigen::VectorXd v = v0;
    Eigen::VectorXd buf(nx * ny);
    
    for (uint32_t i = 1; i < args.nt; ++i) {
        switch (args.system_type) {
            case SystemType::KLEIN_GORDON:
                if (args.method == IntegrationMethod::GAUTSCHI) {
                    KGESolver::step<double>(u, u_past, buf, L, m, dt);
                } else {
                    KGESolverSV::step<double>(u, u_past, buf, L, m, dt);
                }
                break;
            case SystemType::SINE_GORDON:
                if (args.method == IntegrationMethod::GAUTSCHI) {
                    SGESolver::step<double>(u, u_past, buf, L, m, dt);
                } else {
                    SGESolverSV::step<double>(u, u_past, buf, L, m, dt);
                }
                break;
            case SystemType::DOUBLE_SINE_GORDON:
                if (args.method == IntegrationMethod::GAUTSCHI) {
                    SGEDoubleSolver::step<double>(u, u_past, buf, L, m, dt);
                } else {
                    SGEDoubleSolverSV::step<double>(u, u_past, buf, L, m, dt);
                }
                break;
            case SystemType::HYPERBOLIC_SINE_GORDON:
                if (args.method == IntegrationMethod::GAUTSCHI) {
                    SGEHyperbolicSolver::step<double>(u, u_past, buf, L, m, dt);
                } else {
                    SGEHyperbolicSolverSV::step<double>(u, u_past, buf, L, m, dt);
                }
                break;
            case SystemType::PHI4:
                if (args.method == IntegrationMethod::GAUTSCHI) {
                    Phi4Solver::step<double>(u, u_past, buf, L, m, dt);
                } else {
                    Phi4SolverSV::step<double>(u, u_past, buf, L, m, dt);
                }
                break;
            default:
                std::cerr << "Unsupported system type for real-space waves\n";
                exit(1);
        }
        
        neumann_bc_no_velocity<double>(u, nx, ny);
        v = (u - u_past) / dt;
        
        if (i % freq == 0) {
            uint32_t snapshot_idx = i / freq;
            if (snapshot_idx < args.snapshots) {
                u_save_mat.row(snapshot_idx) = u.transpose();
                if (args.velocity_file) {
                    v_save_mat.row(snapshot_idx) = v.transpose();
                }
            }
        }
    }
    
    const std::vector<uint32_t> shape = {args.snapshots, ny, nx};
    save_to_npy(args.trajectory_file, u_save, shape);
    
    if (args.velocity_file) {
        save_to_npy(args.velocity_file.value(), v_save, shape);
    }
}

void run_host_3d(const CommandLineArgs& args) {
    const uint32_t nx = args.n;
    const uint32_t ny = args.n;
    const uint32_t nz = args.n;
    const double dx = 2 * args.L / (nx - 1);
    const double dy = 2 * args.L / (ny - 1);
    const double dz = 2 * args.L / (nz - 1);
    const double dt = args.T / args.nt;
    const auto freq = args.nt / args.snapshots;
    
    std::vector<uint32_t> input_shape;
    Eigen::VectorXd u0 = read_from_npy<double>(args.initial_u_file, input_shape);
    
    if (input_shape.size() != 3 || input_shape[0] != nz || input_shape[1] != ny || input_shape[2] != nx) {
        std::cerr << "Error: Input array dimensions mismatch\n";
        std::cerr << "Expected: " << nz << "x" << ny << "x" << nx << "\n";
        std::cerr << "Got: " << input_shape[0] << "x" << input_shape[1] << "x" << input_shape[2] << "\n";
        exit(1);
    }
    
    Eigen::VectorXd v0;
    if (args.initial_v_file) {
        v0 = read_from_npy<double>(args.initial_v_file.value(), input_shape);
        if (input_shape.size() != 3 || input_shape[0] != nz || input_shape[1] != ny || input_shape[2] != nx) {
            std::cerr << "Error: Velocity array dimensions mismatch\n";
            std::cerr << "Expected: " << nz << "x" << ny << "x" << nx << "\n";
            std::cerr << "Got: " << input_shape[0] << "x" << input_shape[1] << "x" << input_shape[2] << "\n";
            exit(1);
        }
    } else {
        v0 = Eigen::VectorXd::Zero(nx * ny * nz);
    }
    
    Eigen::VectorXd u_past = u0 - dt * v0;
    
    Eigen::VectorXd m = load_or_default_3d<double>(args.focussing_file, nx, ny, nz, 1.0);
    Eigen::VectorXd c = load_or_default_3d<double>(args.anisotropy_file, nx, ny, nz, 1.0);
    
    Eigen::SparseMatrix<double> L;
    if (args.anisotropy_file) {
        L = (-(build_anisotropic_laplacian_noflux_3d<double>(nx - 2, ny - 2, nz - 2, dx, dy, dz, c))).eval();
    } else {
        L = (-(build_laplacian_noflux_3d<double>(nx - 2, ny - 2, nz - 2, dx, dy, dz))).eval();
    }
    
    Eigen::VectorXd u_save(args.snapshots * nx * ny * nz);
    Eigen::VectorXd v_save;
    
    if (args.velocity_file) {
        v_save.resize(args.snapshots * nx * ny * nz);
    }
    
    Eigen::Map<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>> u_save_mat(
        u_save.data(), args.snapshots, nx * ny * nz);
    
    Eigen::Map<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>> v_save_mat;
    if (args.velocity_file) {
        new (&v_save_mat) Eigen::Map<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(
            v_save.data(), args.snapshots, nx * ny * nz);
    }
    
    u_save_mat.row(0) = u0.transpose();
    if (args.velocity_file) {
        v_save_mat.row(0) = v0.transpose();
    }
    
    Eigen::VectorXd u = u0;
    Eigen::VectorXd v = v0;
    Eigen::VectorXd buf(nx * ny * nz);
    
    for (uint32_t i = 1; i < args.nt; ++i) {
        switch (args.system_type) {
            case SystemType::KLEIN_GORDON:
                if (args.method == IntegrationMethod::GAUTSCHI) {
                    KGESolver3D::step<double>(u, u_past, buf, L, m, dt);
                } else {
                    KGESolverSV3D::step<double>(u, u_past, buf, L, m, dt);
                }
                break;
            default:
                std::cerr << "Unsupported 3D system type for real-space waves\n";
                exit(1);
        }
        
        neumann_bc_no_velocity_3d<double>(u, nx, ny, nz);
        v = (u - u_past) / dt;
        
        if (i % freq == 0) {
            uint32_t snapshot_idx = i / freq;
            if (snapshot_idx < args.snapshots) {
                u_save_mat.row(snapshot_idx) = u.transpose();
                if (args.velocity_file) {
                    v_save_mat.row(snapshot_idx) = v.transpose();
                }
            }
        }
    }
    
    const std::vector<uint32_t> shape = {args.snapshots, nz, ny, nx};
    save_to_npy(args.trajectory_file, u_save, shape);
    
    if (args.velocity_file) {
        save_to_npy(args.velocity_file.value(), v_save, shape);
    }
}

#ifdef HAVE_CUDA
void run_device_2d(const CommandLineArgs& args) {
    const uint32_t nx = args.n;
    const uint32_t ny = args.n;
    const double dx = 2 * args.L / (nx - 1);
    const double dy = 2 * args.L / (ny - 1);
    const double dt = args.T / args.nt;
    const auto freq = args.nt / args.snapshots;
    
    std::vector<uint32_t> input_shape;
    Eigen::VectorXd u0 = read_from_npy<double>(args.initial_u_file, input_shape);
    
    if (input_shape.size() != 2 || input_shape[0] != ny || input_shape[1] != nx) {
        std::cerr << "Error: Input array dimensions mismatch\n";
        std::cerr << "Expected: " << ny << "x" << nx << "\n";
        std::cerr << "Got: " << input_shape[0] << "x" << input_shape[1] << "\n";
        exit(1);
    }
    
    Eigen::VectorXd v0;
    if (args.initial_v_file) {
        v0 = read_from_npy<double>(args.initial_v_file.value(), input_shape);
        if (input_shape.size() != 2 || input_shape[0] != ny || input_shape[1] != nx) {
            std::cerr << "Error: Velocity array dimensions mismatch\n";
            std::cerr << "Expected: " << ny << "x" << nx << "\n";
            std::cerr << "Got: " << input_shape[0] << "x" << input_shape[1] << "\n";
            exit(1);
        }
    } else {
        v0 = Eigen::VectorXd::Zero(nx * ny);
    }
    
    Eigen::VectorXd m = load_or_default<double>(args.focussing_file, nx, ny, 1.0);
    Eigen::VectorXd c = load_or_default<double>(args.anisotropy_file, nx, ny, 1.0);
    
    Eigen::SparseMatrix<double> L;
    if (args.anisotropy_file) {
        L = build_anisotropic_laplacian_noflux<double>(nx - 2, ny - 2, dx, dy, c);
    } else {
        L = build_laplacian_noflux<double>(nx - 2, ny - 2, dx, dy);
    }
    
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
    
    Eigen::VectorXd u_save(args.snapshots * nx * ny);
    Eigen::VectorXd v_save;
    
    if (args.velocity_file) {
        v_save.resize(args.snapshots * nx * ny);
    }
    
    device::KGESolverDevice::Parameters params(args.snapshots, freq, 10);
    
    switch (args.system_type) {
        case SystemType::KLEIN_GORDON: {
            device::KGESolverDevice solver(d_row_ptr, d_col_ind, d_values, m.data(),
                                          nx * ny, L.nonZeros(), u0.data(), v0.data(),
                                          dt, params);
            
            for (uint32_t i = 1; i < args.nt; ++i) {
                solver.step();
                solver.apply_bc();
                if (i % freq == 0) {
                    uint32_t snapshot_idx = i / freq;
                    if (snapshot_idx < args.snapshots) {
                        solver.store_snapshot(snapshot_idx);
                    }
                }
            }
            
            if (args.velocity_file) {
                solver.transfer_snapshots(u_save.data(), v_save.data());
            } else {
                solver.transfer_snapshots(u_save.data(), nullptr);
            }
            break;
        }
        case SystemType::SINE_GORDON: {
            device::SGESolverDevice solver(d_row_ptr, d_col_ind, d_values, m.data(),
                                          nx * ny, L.nonZeros(), u0.data(), v0.data(),
                                          dt, params);
            
            for (uint32_t i = 1; i < args.nt; ++i) {
                solver.step();
                solver.apply_bc();
                if (i % freq == 0) {
                    uint32_t snapshot_idx = i / freq;
                    if (snapshot_idx < args.snapshots) {
                        solver.store_snapshot(snapshot_idx);
                    }
                }
            }
            
            if (args.velocity_file) {
                solver.transfer_snapshots(u_save.data(), v_save.data());
            } else {
                solver.transfer_snapshots(u_save.data(), nullptr);
            }
            break;
        }
        case SystemType::DOUBLE_SINE_GORDON: {
            device::SGEDoubleSolverDevice solver(d_row_ptr, d_col_ind, d_values, m.data(),
                                               nx * ny, L.nonZeros(), u0.data(), v0.data(),
                                               dt, params);
            
            for (uint32_t i = 1; i < args.nt; ++i) {
                solver.step();
                solver.apply_bc();
                if (i % freq == 0) {
                    uint32_t snapshot_idx = i / freq;
                    if (snapshot_idx < args.snapshots) {
                        solver.store_snapshot(snapshot_idx);
                    }
                }
            }
            
            if (args.velocity_file) {
                solver.transfer_snapshots(u_save.data(), v_save.data());
            } else {
                solver.transfer_snapshots(u_save.data(), nullptr);
            }
            break;
        }
        case SystemType::HYPERBOLIC_SINE_GORDON: {
            device::SGEHyperbolicSolverDevice solver(d_row_ptr, d_col_ind, d_values, m.data(),
                                                   nx * ny, L.nonZeros(), u0.data(), v0.data(),
                                                   dt, params);
            
            for (uint32_t i = 1; i < args.nt; ++i) {
                solver.step();
                solver.apply_bc();
                if (i % freq == 0) {
                    uint32_t snapshot_idx = i / freq;
                    if (snapshot_idx < args.snapshots) {
                        solver.store_snapshot(snapshot_idx);
                    }
                }
            }
            
            if (args.velocity_file) {
                solver.transfer_snapshots(u_save.data(), v_save.data());
            } else {
                solver.transfer_snapshots(u_save.data(), nullptr);
            }
            break;
        }
        case SystemType::PHI4: {
            device::Phi4SolverDevice solver(d_row_ptr, d_col_ind, d_values, m.data(),
                                          nx * ny, L.nonZeros(), u0.data(), v0.data(),
                                          dt, params);
            
            for (uint32_t i = 1; i < args.nt; ++i) {
                solver.step();
                solver.apply_bc();
                if (i % freq == 0) {
                    uint32_t snapshot_idx = i / freq;
                    if (snapshot_idx < args.snapshots) {
                        solver.store_snapshot(snapshot_idx);
                    }
                }
            }
            
            if (args.velocity_file) {
                solver.transfer_snapshots(u_save.data(), v_save.data());
            } else {
                solver.transfer_snapshots(u_save.data(), nullptr);
            }
            break;
        }
        default:
            std::cerr << "Unsupported system type for real-space waves on device\n";
            exit(1);
    }
    
    const std::vector<uint32_t> shape = {args.snapshots, ny, nx};
    save_to_npy(args.trajectory_file, u_save, shape);
    
    if (args.velocity_file) {
        save_to_npy(args.velocity_file.value(), v_save, shape);
    }
    
    cudaFree(d_row_ptr);
    cudaFree(d_col_ind);
    cudaFree(d_values);
}
#endif

int main(int argc, char **argv) {
    CommandLineArgs args = CommandLineParser::parse_real_wave_args(argc, argv);
    
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
