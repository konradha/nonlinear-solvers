#ifndef SOLVER_FACTORY_UNIFIED_HPP
#define SOLVER_FACTORY_UNIFIED_HPP

#include "cmdline_parser.hpp"

// Forward declarations for host solvers
namespace nlsolvers {
    // Base classes
    template <typename Scalar_t> class SolverBase;
    template <typename Scalar_t> class NLSESolverBase;
    template <typename Scalar_t> class RealSpaceSolverBase;
    
    // NLSE solvers
    template <typename Scalar_t> class NLSECubicSolver;
    template <typename Scalar_t> class NLSECubicSolver3D;
    template <typename Scalar_t> class NLSECubicGautschiSolver;
    template <typename Scalar_t> class NLSECubicGautschiSolver3D;
    template <typename Scalar_t> class NLSECubicQuinticSolver;
    template <typename Scalar_t> class NLSECubicQuinticGautschiSolver;
    template <typename Scalar_t> class NLSESaturatingSolver;
    template <typename Scalar_t> class NLSESaturatingGautschiSolver;
    
    // Real-space solvers
    template <typename Scalar_t> class KGSolver;
    template <typename Scalar_t> class KGSolver3D;
    template <typename Scalar_t> class KGSVSolver;
    template <typename Scalar_t> class KGSVSolver3D;
    template <typename Scalar_t> class SGSingleSolver;
    template <typename Scalar_t> class SGSingleSVSolver;
    template <typename Scalar_t> class SGDoubleSolver;
    template <typename Scalar_t> class SGHyperbolicSolver;
    template <typename Scalar_t> class Phi4Solver;
}

// Forward declarations for device solvers
namespace device {
    // Base classes
    template <typename Scalar_t, int Dim> class SolverBase;
    template <typename Scalar_t, int Dim> class NLSESolverBase;
    template <typename Scalar_t, int Dim> class RealSpaceSolverBase;
    
    // NLSE solvers
    template <typename Scalar_t, int Dim> class NLSECubicSolver;
    template <typename Scalar_t, int Dim> class NLSECubicQuinticSolver;
    template <typename Scalar_t, int Dim> class NLSESaturatingSolver;
    
    // Real-space solvers
    template <typename Scalar_t, int Dim> class KGSolver;
    template <typename Scalar_t, int Dim> class SGSingleSolver;
    template <typename Scalar_t, int Dim> class SGDoubleSolver;
    template <typename Scalar_t, int Dim> class SGHyperbolicSolver;
    template <typename Scalar_t, int Dim> class Phi4Solver;
}

namespace nlsolvers {

class SolverFactory {
public:
    // Create NLSE solver
    template <typename Scalar_t>
    static std::unique_ptr<SolverBase<Scalar_t>> create_nlse_solver(const SolverParams& params) {
        if (params.device_type == SolverParams::DeviceType::HOST) {
            return create_nlse_host_solver<Scalar_t>(params);
        } else {
            return create_nlse_device_solver<Scalar_t>(params);
        }
    }
    
    // Create real-space solver
    template <typename Scalar_t>
    static std::unique_ptr<SolverBase<Scalar_t>> create_realspace_solver(const SolverParams& params) {
        if (params.device_type == SolverParams::DeviceType::HOST) {
            return create_realspace_host_solver<Scalar_t>(params);
        } else {
            return create_realspace_device_solver<Scalar_t>(params);
        }
    }
    
private:
    // Create host NLSE solver
    template <typename Scalar_t>
    static std::unique_ptr<SolverBase<Scalar_t>> create_nlse_host_solver(const SolverParams& params) {
        std::unique_ptr<NLSESolverBase<Scalar_t>> solver;
        
        // Create appropriate solver based on equation type, dimension, and method
        if (params.dimension == 2) {
            if (params.system_type == SolverParams::SystemType::NLSE_CUBIC) {
                if (params.method == SolverParams::Method::SS2) {
                    solver = std::make_unique<NLSECubicSolver<Scalar_t>>();
                } else if (params.method == SolverParams::Method::GAUTSCHI) {
                    solver = std::make_unique<NLSECubicGautschiSolver<Scalar_t>>();
                }
            } else if (params.system_type == SolverParams::SystemType::NLSE_CUBIC_QUINTIC) {
                if (params.method == SolverParams::Method::SS2) {
                    solver = std::make_unique<NLSECubicQuinticSolver<Scalar_t>>(params.sigma1, params.sigma2);
                } else if (params.method == SolverParams::Method::GAUTSCHI) {
                    solver = std::make_unique<NLSECubicQuinticGautschiSolver<Scalar_t>>(params.sigma1, params.sigma2);
                }
            } else if (params.system_type == SolverParams::SystemType::NLSE_SATURATING) {
                if (params.method == SolverParams::Method::SS2) {
                    solver = std::make_unique<NLSESaturatingSolver<Scalar_t>>();
                } else if (params.method == SolverParams::Method::GAUTSCHI) {
                    solver = std::make_unique<NLSESaturatingGautschiSolver<Scalar_t>>();
                }
            }
        } else if (params.dimension == 3) {
            if (params.system_type == SolverParams::SystemType::NLSE_CUBIC) {
                if (params.method == SolverParams::Method::SS2) {
                    solver = std::make_unique<NLSECubicSolver3D<Scalar_t>>();
                } else if (params.method == SolverParams::Method::GAUTSCHI) {
                    solver = std::make_unique<NLSECubicGautschiSolver3D<Scalar_t>>();
                }
            }
            // Add other 3D NLSE solvers as needed
        }
        
        if (!solver) {
            throw std::runtime_error("Unsupported NLSE solver configuration");
        }
        
        return solver;
    }
    
    // Create host real-space solver
    template <typename Scalar_t>
    static std::unique_ptr<SolverBase<Scalar_t>> create_realspace_host_solver(const SolverParams& params) {
        std::unique_ptr<RealSpaceSolverBase<Scalar_t>> solver;
        
        // Create appropriate solver based on equation type, dimension, and method
        if (params.dimension == 2) {
            if (params.system_type == SolverParams::SystemType::KLEIN_GORDON) {
                if (params.method == SolverParams::Method::GAUTSCHI) {
                    solver = std::make_unique<KGSolver<Scalar_t>>();
                } else if (params.method == SolverParams::Method::STORMER_VERLET) {
                    solver = std::make_unique<KGSVSolver<Scalar_t>>();
                }
            } else if (params.system_type == SolverParams::SystemType::SINE_GORDON) {
                if (params.method == SolverParams::Method::GAUTSCHI) {
                    solver = std::make_unique<SGSingleSolver<Scalar_t>>();
                } else if (params.method == SolverParams::Method::STORMER_VERLET) {
                    solver = std::make_unique<SGSingleSVSolver<Scalar_t>>();
                }
            } else if (params.system_type == SolverParams::SystemType::SINE_GORDON_DOUBLE) {
                solver = std::make_unique<SGDoubleSolver<Scalar_t>>(params.alpha, params.beta);
            } else if (params.system_type == SolverParams::SystemType::SINE_GORDON_HYPERBOLIC) {
                solver = std::make_unique<SGHyperbolicSolver<Scalar_t>>();
            } else if (params.system_type == SolverParams::SystemType::PHI4) {
                solver = std::make_unique<Phi4Solver<Scalar_t>>();
            }
        } else if (params.dimension == 3) {
            if (params.system_type == SolverParams::SystemType::KLEIN_GORDON) {
                if (params.method == SolverParams::Method::GAUTSCHI) {
                    solver = std::make_unique<KGSolver3D<Scalar_t>>();
                } else if (params.method == SolverParams::Method::STORMER_VERLET) {
                    solver = std::make_unique<KGSVSolver3D<Scalar_t>>();
                }
            }
            // Add other 3D real-space solvers as needed
        }
        
        if (!solver) {
            throw std::runtime_error("Unsupported real-space solver configuration");
        }
        
        return solver;
    }
    
    // Create device NLSE solver
    template <typename Scalar_t>
    static std::unique_ptr<SolverBase<Scalar_t>> create_nlse_device_solver(const SolverParams& params) {
        // Create a wrapper around device solver
        class DeviceNLSESolverWrapper : public SolverBase<Scalar_t> {
        public:
            template <int Dim>
            DeviceNLSESolverWrapper(std::unique_ptr<device::NLSESolverBase<Scalar_t, Dim>> solver)
                : solver_(std::move(solver)), dimension_(Dim) {}
            
            void initialize(const std::vector<uint32_t>& grid_sizes,
                           const std::vector<Scalar_t>& domain_sizes) override {
                solver_->initialize(grid_sizes, domain_sizes);
            }
            
            void load_initial_condition(const std::string& input_file) override {
                solver_->load_initial_condition(input_file);
            }
            
            void load_anisotropy(const std::string& anisotropy_file) override {
                // Not all device solvers support this yet
            }
            
            void load_focusing(const std::string& focusing_file) override {
                if (dimension_ == 2) {
                    static_cast<device::NLSESolverBase<Scalar_t, 2>*>(solver_.get())->load_focusing_parameter(focusing_file);
                } else if (dimension_ == 3) {
                    static_cast<device::NLSESolverBase<Scalar_t, 3>*>(solver_.get())->load_focusing_parameter(focusing_file);
                }
            }
            
            void run(Scalar_t T, uint32_t nt, uint32_t num_snapshots,
                    const std::string& output_file) override {
                solver_->run(T, nt, num_snapshots, output_file);
            }
            
        private:
            std::unique_ptr<device::SolverBase<Scalar_t, 2>> solver_;
            int dimension_;
        };
        
        // Create device solver based on dimension
        if (params.dimension == 2) {
            // Convert params to device params
            device::SolverParams device_params;
            device_params.grid_sizes = params.grid_sizes;
            device_params.domain_sizes = params.domain_sizes;
            device_params.T = params.T;
            device_params.nt = params.nt;
            device_params.num_snapshots = params.num_snapshots;
            device_params.input_file = params.input_file;
            device_params.output_file = params.output_file;
            if (params.focusing_file) {
                device_params.m_file = *params.focusing_file;
            }
            
            // Set equation type
            if (params.system_type == SolverParams::SystemType::NLSE_CUBIC) {
                device_params.equation_type = device::SolverParams::EquationType::NLSE_CUBIC;
            } else if (params.system_type == SolverParams::SystemType::NLSE_CUBIC_QUINTIC) {
                device_params.equation_type = device::SolverParams::EquationType::NLSE_CUBIC_QUINTIC;
                device_params.sigma1 = params.sigma1;
                device_params.sigma2 = params.sigma2;
            } else if (params.system_type == SolverParams::SystemType::NLSE_SATURATING) {
                device_params.equation_type = device::SolverParams::EquationType::NLSE_SATURATING;
            }
            
            // Set method
            if (params.method == SolverParams::Method::SS2) {
                device_params.method = device::SolverParams::Method::SS2;
            } else if (params.method == SolverParams::Method::SEWI) {
                device_params.method = device::SolverParams::Method::SEWI;
            }
            
            // Create device solver
            auto device_solver = device::SolverFactory::create_nlse_solver_2d<Scalar_t>(device_params);
            return std::make_unique<DeviceNLSESolverWrapper<2>>(std::move(device_solver));
        } else if (params.dimension == 3) {
            // Similar implementation for 3D
            // ...
        }
        
        throw std::runtime_error("Unsupported NLSE device solver configuration");
    }
    
    // Create device real-space solver
    template <typename Scalar_t>
    static std::unique_ptr<SolverBase<Scalar_t>> create_realspace_device_solver(const SolverParams& params) {
        // Create a wrapper around device solver
        class DeviceRealSpaceSolverWrapper : public SolverBase<Scalar_t> {
        public:
            template <int Dim>
            DeviceRealSpaceSolverWrapper(std::unique_ptr<device::RealSpaceSolverBase<Scalar_t, Dim>> solver)
                : solver_(std::move(solver)), dimension_(Dim) {}
            
            void initialize(const std::vector<uint32_t>& grid_sizes,
                           const std::vector<Scalar_t>& domain_sizes) override {
                solver_->initialize(grid_sizes, domain_sizes);
            }
            
            void load_initial_condition(const std::string& input_file) override {
                solver_->load_initial_condition(input_file);
            }
            
            void load_initial_velocity(const std::string& velocity_file) override {
                if (dimension_ == 2) {
                    static_cast<device::RealSpaceSolverBase<Scalar_t, 2>*>(solver_.get())->load_initial_velocity(velocity_file);
                } else if (dimension_ == 3) {
                    static_cast<device::RealSpaceSolverBase<Scalar_t, 3>*>(solver_.get())->load_initial_velocity(velocity_file);
                }
            }
            
            void load_anisotropy(const std::string& anisotropy_file) override {
                // Not all device solvers support this yet
            }
            
            void load_coupling(const std::string& coupling_file) override {
                if (dimension_ == 2) {
                    static_cast<device::RealSpaceSolverBase<Scalar_t, 2>*>(solver_.get())->load_coupling_parameter(coupling_file);
                } else if (dimension_ == 3) {
                    static_cast<device::RealSpaceSolverBase<Scalar_t, 3>*>(solver_.get())->load_coupling_parameter(coupling_file);
                }
            }
            
            void set_velocity_output_file(const std::string& velocity_output_file) override {
                if (dimension_ == 2) {
                    static_cast<device::RealSpaceSolverBase<Scalar_t, 2>*>(solver_.get())->set_velocity_output_file(velocity_output_file);
                } else if (dimension_ == 3) {
                    static_cast<device::RealSpaceSolverBase<Scalar_t, 3>*>(solver_.get())->set_velocity_output_file(velocity_output_file);
                }
            }
            
            void run(Scalar_t T, uint32_t nt, uint32_t num_snapshots,
                    const std::string& output_file) override {
                solver_->run(T, nt, num_snapshots, output_file);
            }
            
        private:
            std::unique_ptr<device::SolverBase<Scalar_t, 2>> solver_;
            int dimension_;
        };
        
        // Create device solver based on dimension
        if (params.dimension == 2) {
            // Convert params to device params
            device::SolverParams device_params;
            device_params.grid_sizes = params.grid_sizes;
            device_params.domain_sizes = params.domain_sizes;
            device_params.T = params.T;
            device_params.nt = params.nt;
            device_params.num_snapshots = params.num_snapshots;
            device_params.input_file = params.input_file;
            device_params.output_file = params.output_file;
            if (params.focusing_file) {
                device_params.m_file = *params.focusing_file;
            }
            if (params.initial_velocity_file) {
                device_params.velocity_file = *params.initial_velocity_file;
            }
            if (params.velocity_output_file) {
                device_params.velocity_output_file = *params.velocity_output_file;
            }
            
            // Set equation type
            if (params.system_type == SolverParams::SystemType::KLEIN_GORDON) {
                device_params.equation_type = device::SolverParams::EquationType::KG;
            } else if (params.system_type == SolverParams::SystemType::SINE_GORDON) {
                device_params.equation_type = device::SolverParams::EquationType::SG_SINGLE;
            } else if (params.system_type == SolverParams::SystemType::SINE_GORDON_DOUBLE) {
                device_params.equation_type = device::SolverParams::EquationType::SG_DOUBLE;
            } else if (params.system_type == SolverParams::SystemType::SINE_GORDON_HYPERBOLIC) {
                device_params.equation_type = device::SolverParams::EquationType::SG_HYPERBOLIC;
            } else if (params.system_type == SolverParams::SystemType::PHI4) {
                device_params.equation_type = device::SolverParams::EquationType::PHI4;
            }
            
            // Set method
            if (params.method == SolverParams::Method::GAUTSCHI) {
                device_params.method = device::SolverParams::Method::GAUTSCHI;
            } else if (params.method == SolverParams::Method::STORMER_VERLET) {
                device_params.method = device::SolverParams::Method::STORMER_VERLET;
            }
            
            // Create device solver
            auto device_solver = device::SolverFactory::create_realspace_solver_2d<Scalar_t>(device_params);
            return std::make_unique<DeviceRealSpaceSolverWrapper<2>>(std::move(device_solver));
        } else if (params.dimension == 3) {
            // Similar implementation for 3D
            // ...
        }
        
        throw std::runtime_error("Unsupported real-space device solver configuration");
    }
};

} // namespace nlsolvers

#endif // SOLVER_FACTORY_UNIFIED_HPP
