#ifndef CMDLINE_PARSER_UNIFIED_HPP
#define CMDLINE_PARSER_UNIFIED_HPP

#include <string>
#include <vector>
#include <optional>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <unordered_map>

namespace nlsolvers {

struct SolverParams {
    // Common parameters
    std::vector<uint32_t> grid_sizes;
    std::vector<double> domain_sizes;
    double T;
    uint32_t nt;
    uint32_t num_snapshots;
    std::string input_file;
    std::string output_file;
    std::optional<std::string> anisotropy_file;
    std::optional<std::string> focusing_file;
    
    // Device type
    enum class DeviceType {
        HOST,
        CUDA
    } device_type = DeviceType::HOST;
    
    // System type
    enum class SystemType {
        // NLSE types
        NLSE_CUBIC,
        NLSE_CUBIC_QUINTIC,
        NLSE_SATURATING,
        
        // Real-space types
        KLEIN_GORDON,
        SINE_GORDON,
        SINE_GORDON_DOUBLE,
        SINE_GORDON_HYPERBOLIC,
        PHI4
    } system_type;
    
    // Dimension
    int dimension;
    
    // Method
    enum class Method {
        // NLSE methods
        SS2,
        SEWI,
        
        // Real-space methods
        GAUTSCHI,
        STORMER_VERLET
    } method;
    
    // Additional parameters for real-space equations
    std::optional<std::string> initial_velocity_file;
    std::optional<std::string> velocity_output_file;
    
    // Additional parameters for specific equations
    double sigma1 = 1.0;  // For cubic-quintic NLSE
    double sigma2 = 1.0;  // For cubic-quintic NLSE
    double alpha = 0.5;   // For double sine-Gordon
    double beta = 1.0;    // For double sine-Gordon
};

class CommandLineParser {
public:
    static SolverParams parse(int argc, char** argv) {
        SolverParams params;
        
        if (argc < 2) {
            print_usage(argv[0]);
            throw std::runtime_error("Insufficient command line arguments");
        }
        
        // Parse arguments into a map for easier access
        std::unordered_map<std::string, std::string> args;
        std::vector<std::string> positional_args;
        
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            
            if (arg.substr(0, 2) == "--") {
                size_t pos = arg.find('=');
                if (pos != std::string::npos) {
                    std::string key = arg.substr(2, pos - 2);
                    std::string value = arg.substr(pos + 1);
                    args[key] = value;
                } else {
                    args[arg.substr(2)] = "true";
                }
            } else {
                positional_args.push_back(arg);
            }
        }
        
        // Parse device type
        if (args.count("device")) {
            std::string device = args["device"];
            if (device == "host") {
                params.device_type = SolverParams::DeviceType::HOST;
            } else if (device == "cuda") {
                params.device_type = SolverParams::DeviceType::CUDA;
            } else {
                throw std::runtime_error("Invalid device type: " + device);
            }
        }
        
        // Parse system type
        if (!args.count("system-type")) {
            throw std::runtime_error("Missing required parameter: --system-type");
        }
        
        std::string system_type = args["system-type"];
        if (system_type == "nlse-cubic" || system_type == "cubic") {
            params.system_type = SolverParams::SystemType::NLSE_CUBIC;
        } else if (system_type == "nlse-cubic-quintic" || system_type == "cubic-quintic") {
            params.system_type = SolverParams::SystemType::NLSE_CUBIC_QUINTIC;
        } else if (system_type == "nlse-saturating" || system_type == "saturating") {
            params.system_type = SolverParams::SystemType::NLSE_SATURATING;
        } else if (system_type == "klein-gordon" || system_type == "kg") {
            params.system_type = SolverParams::SystemType::KLEIN_GORDON;
        } else if (system_type == "sine-gordon" || system_type == "sg") {
            params.system_type = SolverParams::SystemType::SINE_GORDON;
        } else if (system_type == "sine-gordon-double" || system_type == "sg-double") {
            params.system_type = SolverParams::SystemType::SINE_GORDON_DOUBLE;
        } else if (system_type == "sine-gordon-hyperbolic" || system_type == "sg-hyperbolic") {
            params.system_type = SolverParams::SystemType::SINE_GORDON_HYPERBOLIC;
        } else if (system_type == "phi4") {
            params.system_type = SolverParams::SystemType::PHI4;
        } else {
            throw std::runtime_error("Invalid system type: " + system_type);
        }
        
        // Parse dimension
        if (!args.count("dim")) {
            throw std::runtime_error("Missing required parameter: --dim");
        }
        
        params.dimension = std::stoi(args["dim"]);
        if (params.dimension != 2 && params.dimension != 3) {
            throw std::runtime_error("Dimension must be 2 or 3");
        }
        
        // Parse method
        if (args.count("method")) {
            std::string method = args["method"];
            if (method == "ss2") {
                params.method = SolverParams::Method::SS2;
            } else if (method == "sewi") {
                params.method = SolverParams::Method::SEWI;
            } else if (method == "gautschi") {
                params.method = SolverParams::Method::GAUTSCHI;
            } else if (method == "stormer-verlet" || method == "sv") {
                params.method = SolverParams::Method::STORMER_VERLET;
            } else {
                throw std::runtime_error("Invalid method: " + method);
            }
        } else {
            // Default methods based on system type
            if (is_nlse_type(params.system_type)) {
                params.method = SolverParams::Method::SS2;
            } else {
                params.method = SolverParams::Method::GAUTSCHI;
            }
        }
        
        // Parse grid and domain sizes
        if (args.count("n")) {
            uint32_t n = std::stoul(args["n"]);
            params.grid_sizes.resize(params.dimension, n);
        } else {
            throw std::runtime_error("Missing required parameter: --n");
        }
        
        if (args.count("L")) {
            double L = std::stod(args["L"]);
            params.domain_sizes.resize(params.dimension, L);
        } else {
            throw std::runtime_error("Missing required parameter: --L");
        }
        
        // Parse time parameters
        if (args.count("T")) {
            params.T = std::stod(args["T"]);
        } else {
            throw std::runtime_error("Missing required parameter: --T");
        }
        
        if (args.count("nt")) {
            params.nt = std::stoul(args["nt"]);
        } else {
            throw std::runtime_error("Missing required parameter: --nt");
        }
        
        if (args.count("snapshots")) {
            params.num_snapshots = std::stoul(args["snapshots"]);
        } else {
            throw std::runtime_error("Missing required parameter: --snapshots");
        }
        
        // Parse file parameters
        if (args.count("initial-u")) {
            params.input_file = args["initial-u"];
        } else {
            throw std::runtime_error("Missing required parameter: --initial-u");
        }
        
        if (args.count("trajectory-file")) {
            params.output_file = args["trajectory-file"];
        } else {
            throw std::runtime_error("Missing required parameter: --trajectory-file");
        }
        
        // Optional parameters
        if (args.count("anisotropy")) {
            params.anisotropy_file = args["anisotropy"];
        }
        
        if (args.count("focussing")) {
            params.focusing_file = args["focussing"];
        }
        
        // Real-space specific parameters
        if (!is_nlse_type(params.system_type)) {
            if (args.count("initial-v")) {
                params.initial_velocity_file = args["initial-v"];
            }
            
            if (args.count("velocity-file")) {
                params.velocity_output_file = args["velocity-file"];
            }
        }
        
        // Equation-specific parameters
        if (params.system_type == SolverParams::SystemType::NLSE_CUBIC_QUINTIC) {
            if (args.count("sigma1")) {
                params.sigma1 = std::stod(args["sigma1"]);
            }
            
            if (args.count("sigma2")) {
                params.sigma2 = std::stod(args["sigma2"]);
            }
        } else if (params.system_type == SolverParams::SystemType::SINE_GORDON_DOUBLE) {
            if (args.count("alpha")) {
                params.alpha = std::stod(args["alpha"]);
            }
            
            if (args.count("beta")) {
                params.beta = std::stod(args["beta"]);
            }
        }
        
        return params;
    }
    
private:
    static void print_usage(const char* program_name) {
        std::cerr << "Usage: " << program_name << " [options]\n\n"
                  << "Required options:\n"
                  << "  --system-type=<type>    System type (nlse-cubic, nlse-cubic-quintic, nlse-saturating,\n"
                  << "                          klein-gordon, sine-gordon, sine-gordon-double,\n"
                  << "                          sine-gordon-hyperbolic, phi4)\n"
                  << "  --dim=<dim>             Dimension (2 or 3)\n"
                  << "  --n=<n>                 Grid size in each dimension\n"
                  << "  --L=<L>                 Domain size in each dimension\n"
                  << "  --T=<T>                 Total simulation time\n"
                  << "  --nt=<nt>               Number of time steps\n"
                  << "  --snapshots=<n>         Number of snapshots to save\n"
                  << "  --initial-u=<file>      Initial condition file (NPY format)\n"
                  << "  --trajectory-file=<file> Output trajectory file (NPY format)\n\n"
                  << "Optional options:\n"
                  << "  --device=<type>         Device type (host or cuda, default: host)\n"
                  << "  --method=<method>       Integration method (ss2, sewi, gautschi, sv)\n"
                  << "  --anisotropy=<file>     Anisotropy coefficient file (NPY format)\n"
                  << "  --focussing=<file>      Focusing/coupling parameter file (NPY format)\n"
                  << "  --initial-v=<file>      Initial velocity file for real-space equations (NPY format)\n"
                  << "  --velocity-file=<file>  Output velocity file for real-space equations (NPY format)\n"
                  << "  --sigma1=<value>        Sigma1 parameter for cubic-quintic NLSE (default: 1.0)\n"
                  << "  --sigma2=<value>        Sigma2 parameter for cubic-quintic NLSE (default: 1.0)\n"
                  << "  --alpha=<value>         Alpha parameter for double sine-Gordon (default: 0.5)\n"
                  << "  --beta=<value>          Beta parameter for double sine-Gordon (default: 1.0)\n\n"
                  << "Examples:\n"
                  << "  " << program_name << " --device=host --system-type=klein-gordon --dim=3 --anisotropy=c_file.npy\n"
                  << "     --focussing=m_file.npy --L=10. --n=200 --T=10. --nt=1000 --snapshots=32 --method=gautschi\n"
                  << "     --initial-u=initial.npy --initial-v=initial_velocity.npy --trajectory-file=traj.npy\n"
                  << "     --velocity-file=vel.npy\n\n"
                  << "  " << program_name << " --device=cuda --system-type=sine-gordon --dim=2 --L=10. --n=200 --T=10.\n"
                  << "     --nt=1000 --snapshots=32 --initial-u=initial.npy --trajectory-file=traj.npy\n";
    }
    
    static bool is_nlse_type(SolverParams::SystemType type) {
        return type == SolverParams::SystemType::NLSE_CUBIC ||
               type == SolverParams::SystemType::NLSE_CUBIC_QUINTIC ||
               type == SolverParams::SystemType::NLSE_SATURATING;
    }
};

} // namespace nlsolvers

#endif // CMDLINE_PARSER_UNIFIED_HPP
