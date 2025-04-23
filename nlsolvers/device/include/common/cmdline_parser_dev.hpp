#ifndef CMDLINE_PARSER_DEV_HPP
#define CMDLINE_PARSER_DEV_HPP

#include <string>
#include <vector>
#include <optional>
#include <stdexcept>
#include <iostream>

namespace device {

struct SolverParams {
    // Common parameters
    std::vector<uint32_t> grid_sizes;
    std::vector<double> domain_sizes;
    double T;
    uint32_t nt;
    uint32_t num_snapshots;
    std::string input_file;
    std::string output_file;
    std::optional<std::string> m_file;
    
    // Equation type
    enum class EquationType {
        NLSE_CUBIC,
        NLSE_CUBIC_QUINTIC,
        NLSE_SATURATING,
        KG,
        SG_SINGLE,
        SG_DOUBLE,
        SG_HYPERBOLIC,
        PHI4
    } equation_type;
    
    // Dimension
    int dimension;
    
    // Method
    enum class Method {
        SS2,
        SEWI,
        GAUTSCHI,
        STORMER_VERLET
    } method;
    
    // Additional parameters for real-space equations
    std::optional<std::string> velocity_file;
    std::optional<std::string> velocity_output_file;
    
    // Additional parameters for NLSE cubic-quintic
    double sigma1 = 1.0;
    double sigma2 = 1.0;
};

class CommandLineParser {
public:
    static SolverParams parse_nlse(int argc, char** argv) {
        SolverParams params;
        
        if (argc < 10) {
            std::cerr << "Usage: " << argv[0]
                      << " --equation=<type> --dim=<dim> --method=<method> nx ny Lx Ly input_u0.npy output_traj.npy T nt num_snapshots [input_m.npy] [sigma1 sigma2]" << std::endl;
            std::cerr << "Example: " << argv[0]
                      << " --equation=cubic --dim=2 --method=ss2 256 256 10.0 10.0 initial.npy evolution.npy 1.5 500 100" << std::endl;
            std::cerr << "Example with m(x,y): " << argv[0]
                      << " --equation=cubic --dim=2 --method=sewi 256 256 10.0 10.0 initial.npy evolution.npy 1.5 500 100 focusing.npy" << std::endl;
            std::cerr << "Example with cubic-quintic: " << argv[0]
                      << " --equation=cubic_quintic --dim=2 --method=ss2 256 256 10.0 10.0 initial.npy evolution.npy 1.5 500 100 focusing.npy 1.0 0.5" << std::endl;
            throw std::runtime_error("Invalid command line arguments");
        }
        
        int arg_offset = 0;
        
        // Parse equation type
        std::string eq_arg = argv[1];
        if (eq_arg.find("--equation=") != 0) {
            std::cerr << "Error: First argument must be --equation=<type>" << std::endl;
            throw std::runtime_error("Invalid equation type");
        }
        std::string eq_type = eq_arg.substr(11);
        if (eq_type == "cubic") {
            params.equation_type = SolverParams::EquationType::NLSE_CUBIC;
        } else if (eq_type == "cubic_quintic") {
            params.equation_type = SolverParams::EquationType::NLSE_CUBIC_QUINTIC;
        } else if (eq_type == "saturating") {
            params.equation_type = SolverParams::EquationType::NLSE_SATURATING;
        } else {
            std::cerr << "Error: Unknown NLSE equation type: " << eq_type << std::endl;
            throw std::runtime_error("Invalid equation type");
        }
        arg_offset++;
        
        // Parse dimension
        std::string dim_arg = argv[2];
        if (dim_arg.find("--dim=") != 0) {
            std::cerr << "Error: Second argument must be --dim=<dim>" << std::endl;
            throw std::runtime_error("Invalid dimension");
        }
        params.dimension = std::stoi(dim_arg.substr(6));
        if (params.dimension != 2 && params.dimension != 3) {
            std::cerr << "Error: Dimension must be 2 or 3" << std::endl;
            throw std::runtime_error("Invalid dimension");
        }
        arg_offset++;
        
        // Parse method
        std::string method_arg = argv[3];
        if (method_arg.find("--method=") != 0) {
            std::cerr << "Error: Third argument must be --method=<method>" << std::endl;
            throw std::runtime_error("Invalid method");
        }
        std::string method_type = method_arg.substr(9);
        if (method_type == "ss2") {
            params.method = SolverParams::Method::SS2;
        } else if (method_type == "sewi") {
            params.method = SolverParams::Method::SEWI;
        } else {
            std::cerr << "Error: Unknown NLSE method: " << method_type << std::endl;
            throw std::runtime_error("Invalid method");
        }
        arg_offset++;
        
        // Parse grid sizes and domain sizes
        params.grid_sizes.resize(params.dimension);
        params.domain_sizes.resize(params.dimension);
        
        for (int i = 0; i < params.dimension; i++) {
            params.grid_sizes[i] = std::stoul(argv[i + 1 + arg_offset]);
            params.domain_sizes[i] = std::stod(argv[i + 1 + params.dimension + arg_offset]);
        }
        
        // Parse remaining parameters
        int base_idx = 1 + arg_offset + 2 * params.dimension;
        params.input_file = argv[base_idx];
        params.output_file = argv[base_idx + 1];
        params.T = std::stod(argv[base_idx + 2]);
        params.nt = std::stoul(argv[base_idx + 3]);
        params.num_snapshots = std::stoul(argv[base_idx + 4]);
        
        // Optional m file
        if (argc > base_idx + 5) {
            params.m_file = argv[base_idx + 5];
        }
        
        // Optional sigma1 and sigma2 for cubic-quintic
        if (params.equation_type == SolverParams::EquationType::NLSE_CUBIC_QUINTIC && argc > base_idx + 6) {
            params.sigma1 = std::stod(argv[base_idx + 6]);
            if (argc > base_idx + 7) {
                params.sigma2 = std::stod(argv[base_idx + 7]);
            }
        }
        
        return params;
    }
    
    static SolverParams parse_realspace(int argc, char** argv) {
        SolverParams params;
        
        if (argc < 12) {
            std::cerr << "Usage: " << argv[0]
                      << " --equation=<type> --dim=<dim> --method=<method> nx ny Lx Ly input_u0.npy input_v0.npy output_traj.npy output_vel.npy T nt num_snapshots [input_m.npy]" << std::endl;
            std::cerr << "Example: " << argv[0]
                      << " --equation=kg --dim=2 --method=gautschi 256 256 10.0 10.0 initial.npy velocity.npy evolution_u.npy evolution_v.npy 1.5 500 100" << std::endl;
            std::cerr << "Example with m(x,y): " << argv[0]
                      << " --equation=kg --dim=2 --method=stormer_verlet 256 256 10.0 10.0 initial.npy velocity.npy evolution_u.npy evolution_v.npy 1.5 500 100 coupling.npy" << std::endl;
            throw std::runtime_error("Invalid command line arguments");
        }
        
        int arg_offset = 0;
        
        // Parse equation type
        std::string eq_arg = argv[1];
        if (eq_arg.find("--equation=") != 0) {
            std::cerr << "Error: First argument must be --equation=<type>" << std::endl;
            throw std::runtime_error("Invalid equation type");
        }
        std::string eq_type = eq_arg.substr(11);
        if (eq_type == "kg") {
            params.equation_type = SolverParams::EquationType::KG;
        } else if (eq_type == "sg_single") {
            params.equation_type = SolverParams::EquationType::SG_SINGLE;
        } else if (eq_type == "sg_double") {
            params.equation_type = SolverParams::EquationType::SG_DOUBLE;
        } else if (eq_type == "sg_hyperbolic") {
            params.equation_type = SolverParams::EquationType::SG_HYPERBOLIC;
        } else if (eq_type == "phi4") {
            params.equation_type = SolverParams::EquationType::PHI4;
        } else {
            std::cerr << "Error: Unknown real-space equation type: " << eq_type << std::endl;
            throw std::runtime_error("Invalid equation type");
        }
        arg_offset++;
        
        // Parse dimension
        std::string dim_arg = argv[2];
        if (dim_arg.find("--dim=") != 0) {
            std::cerr << "Error: Second argument must be --dim=<dim>" << std::endl;
            throw std::runtime_error("Invalid dimension");
        }
        params.dimension = std::stoi(dim_arg.substr(6));
        if (params.dimension != 2 && params.dimension != 3) {
            std::cerr << "Error: Dimension must be 2 or 3" << std::endl;
            throw std::runtime_error("Invalid dimension");
        }
        arg_offset++;
        
        // Parse method
        std::string method_arg = argv[3];
        if (method_arg.find("--method=") != 0) {
            std::cerr << "Error: Third argument must be --method=<method>" << std::endl;
            throw std::runtime_error("Invalid method");
        }
        std::string method_type = method_arg.substr(9);
        if (method_type == "gautschi") {
            params.method = SolverParams::Method::GAUTSCHI;
        } else if (method_type == "stormer_verlet" || method_type == "sv") {
            params.method = SolverParams::Method::STORMER_VERLET;
        } else {
            std::cerr << "Error: Unknown real-space method: " << method_type << std::endl;
            throw std::runtime_error("Invalid method");
        }
        arg_offset++;
        
        // Parse grid sizes and domain sizes
        params.grid_sizes.resize(params.dimension);
        params.domain_sizes.resize(params.dimension);
        
        for (int i = 0; i < params.dimension; i++) {
            params.grid_sizes[i] = std::stoul(argv[i + 1 + arg_offset]);
            params.domain_sizes[i] = std::stod(argv[i + 1 + params.dimension + arg_offset]);
        }
        
        // Parse remaining parameters
        int base_idx = 1 + arg_offset + 2 * params.dimension;
        params.input_file = argv[base_idx];
        params.velocity_file = argv[base_idx + 1];
        params.output_file = argv[base_idx + 2];
        params.velocity_output_file = argv[base_idx + 3];
        params.T = std::stod(argv[base_idx + 4]);
        params.nt = std::stoul(argv[base_idx + 5]);
        params.num_snapshots = std::stoul(argv[base_idx + 6]);
        
        // Optional m file
        if (argc > base_idx + 7) {
            params.m_file = argv[base_idx + 7];
        }
        
        return params;
    }
};

} // namespace device

#endif // CMDLINE_PARSER_DEV_HPP
