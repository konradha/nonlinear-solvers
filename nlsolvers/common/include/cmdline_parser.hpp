#ifndef CMDLINE_PARSER_HPP
#define CMDLINE_PARSER_HPP

#include <getopt.h>
#include <string>
#include <optional>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <functional>

enum class SystemType {
    KLEIN_GORDON,
    SINE_GORDON,
    DOUBLE_SINE_GORDON,
    HYPERBOLIC_SINE_GORDON,
    PHI4,
    NLSE_CUBIC,
    NLSE_CUBIC_QUINTIC,
    NLSE_SATURABLE
};

enum class DeviceType {
    HOST,
    CUDA
};

enum class IntegrationMethod {
    GAUTSCHI,
    STRANG_SPLITTING
};

struct CommandLineArgs {
    DeviceType device = DeviceType::HOST;
    SystemType system_type = SystemType::KLEIN_GORDON;
    int dim = 2;
    std::optional<std::string> anisotropy_file;
    std::optional<std::string> focussing_file;
    double L = 10.0;
    int n = 200;
    double T = 10.0;
    int nt = 1000;
    int snapshots = 32;
    IntegrationMethod method = IntegrationMethod::GAUTSCHI;
    std::string initial_u_file;
    std::optional<std::string> initial_v_file;
    std::string trajectory_file;
    std::optional<std::string> velocity_file;
    
    std::unordered_map<std::string, double> system_params;
};

class CommandLineParser {
public:
    static CommandLineArgs parse_real_wave_args(int argc, char** argv) {
        CommandLineArgs args;
        
        static struct option long_options[] = {
            {"device", required_argument, 0, 'd'},
            {"system-type", required_argument, 0, 's'},
            {"dim", required_argument, 0, 'D'},
            {"anisotropy", required_argument, 0, 'a'},
            {"focussing", required_argument, 0, 'f'},
            {"L", required_argument, 0, 'L'},
            {"n", required_argument, 0, 'n'},
            {"T", required_argument, 0, 'T'},
            {"nt", required_argument, 0, 't'},
            {"snapshots", required_argument, 0, 'S'},
            {"method", required_argument, 0, 'm'},
            {"initial-u", required_argument, 0, 'u'},
            {"initial-v", required_argument, 0, 'v'},
            {"trajectory-file", required_argument, 0, 'o'},
            {"velocity-file", required_argument, 0, 'V'},
            {"param", required_argument, 0, 'p'},
            {"help", no_argument, 0, 'h'},
            {0, 0, 0, 0}
        };
        
        int option_index = 0;
        int c;
        
        while ((c = getopt_long(argc, argv, "d:s:D:a:f:L:n:T:t:S:m:u:v:o:V:p:h", 
                               long_options, &option_index)) != -1) {
            switch (c) {
                case 'd':
                    if (std::string(optarg) == "host") {
                        args.device = DeviceType::HOST;
                    } else if (std::string(optarg) == "cuda") {
                        args.device = DeviceType::CUDA;
                    } else {
                        std::cerr << "Invalid device type: " << optarg << std::endl;
                        print_real_wave_usage(argv[0]);
                        exit(1);
                    }
                    break;
                case 's':
                    if (std::string(optarg) == "klein-gordon") {
                        args.system_type = SystemType::KLEIN_GORDON;
                    } else if (std::string(optarg) == "sine-gordon") {
                        args.system_type = SystemType::SINE_GORDON;
                    } else if (std::string(optarg) == "double-sine-gordon") {
                        args.system_type = SystemType::DOUBLE_SINE_GORDON;
                    } else if (std::string(optarg) == "hyperbolic-sine-gordon") {
                        args.system_type = SystemType::HYPERBOLIC_SINE_GORDON;
                    } else if (std::string(optarg) == "phi4") {
                        args.system_type = SystemType::PHI4;
                    } else {
                        std::cerr << "Invalid system type: " << optarg << std::endl;
                        print_real_wave_usage(argv[0]);
                        exit(1);
                    }
                    break;
                case 'D':
                    args.dim = std::stoi(optarg);
                    if (args.dim != 2 && args.dim != 3) {
                        std::cerr << "Dimension must be 2 or 3" << std::endl;
                        print_real_wave_usage(argv[0]);
                        exit(1);
                    }
                    break;
                case 'a':
                    args.anisotropy_file = std::string(optarg);
                    break;
                case 'f':
                    args.focussing_file = std::string(optarg);
                    break;
                case 'L':
                    args.L = std::stod(optarg);
                    break;
                case 'n':
                    args.n = std::stoi(optarg);
                    break;
                case 'T':
                    args.T = std::stod(optarg);
                    break;
                case 't':
                    args.nt = std::stoi(optarg);
                    break;
                case 'S':
                    args.snapshots = std::stoi(optarg);
                    break;
                case 'm':
                    if (std::string(optarg) == "gautschi") {
                        args.method = IntegrationMethod::GAUTSCHI;
                    } else if (std::string(optarg) == "strang") {
                        args.method = IntegrationMethod::STRANG_SPLITTING;
                    } else {
                        std::cerr << "Invalid method: " << optarg << std::endl;
                        print_real_wave_usage(argv[0]);
                        exit(1);
                    }
                    break;
                case 'u':
                    args.initial_u_file = std::string(optarg);
                    break;
                case 'v':
                    args.initial_v_file = std::string(optarg);
                    break;
                case 'o':
                    args.trajectory_file = std::string(optarg);
                    break;
                case 'V':
                    args.velocity_file = std::string(optarg);
                    break;
                case 'p': {
                    std::string param_str(optarg);
                    size_t pos = param_str.find('=');
                    if (pos != std::string::npos) {
                        std::string key = param_str.substr(0, pos);
                        double value = std::stod(param_str.substr(pos + 1));
                        args.system_params[key] = value;
                    }
                    break;
                }
                case 'h':
                    print_real_wave_usage(argv[0]);
                    exit(0);
                    break;
                default:
                    print_real_wave_usage(argv[0]);
                    exit(1);
            }
        }
        
        if (args.initial_u_file.empty()) {
            std::cerr << "Error: initial-u file is required" << std::endl;
            print_real_wave_usage(argv[0]);
            exit(1);
        }
        
        if (args.trajectory_file.empty()) {
            std::cerr << "Error: trajectory-file is required" << std::endl;
            print_real_wave_usage(argv[0]);
            exit(1);
        }
        
        return args;
    }
    
    static CommandLineArgs parse_nlse_args(int argc, char** argv) {
        CommandLineArgs args;
        
        static struct option long_options[] = {
            {"device", required_argument, 0, 'd'},
            {"system-type", required_argument, 0, 's'},
            {"dim", required_argument, 0, 'D'},
            {"anisotropy", required_argument, 0, 'a'},
            {"focussing", required_argument, 0, 'f'},
            {"L", required_argument, 0, 'L'},
            {"n", required_argument, 0, 'n'},
            {"T", required_argument, 0, 'T'},
            {"nt", required_argument, 0, 't'},
            {"snapshots", required_argument, 0, 'S'},
            {"method", required_argument, 0, 'm'},
            {"initial-u", required_argument, 0, 'u'},
            {"trajectory-file", required_argument, 0, 'o'},
            {"param", required_argument, 0, 'p'},
            {"help", no_argument, 0, 'h'},
            {0, 0, 0, 0}
        };
        
        int option_index = 0;
        int c;
        
        while ((c = getopt_long(argc, argv, "d:s:D:a:f:L:n:T:t:S:m:u:o:p:h", 
                               long_options, &option_index)) != -1) {
            switch (c) {
                case 'd':
                    if (std::string(optarg) == "host") {
                        args.device = DeviceType::HOST;
                    } else if (std::string(optarg) == "cuda") {
                        args.device = DeviceType::CUDA;
                    } else {
                        std::cerr << "Invalid device type: " << optarg << std::endl;
                        print_nlse_usage(argv[0]);
                        exit(1);
                    }
                    break;
                case 's':
                    if (std::string(optarg) == "cubic") {
                        args.system_type = SystemType::NLSE_CUBIC;
                    } else if (std::string(optarg) == "cubic-quintic") {
                        args.system_type = SystemType::NLSE_CUBIC_QUINTIC;
                    } else if (std::string(optarg) == "saturable") {
                        args.system_type = SystemType::NLSE_SATURABLE;
                    } else {
                        std::cerr << "Invalid system type: " << optarg << std::endl;
                        print_nlse_usage(argv[0]);
                        exit(1);
                    }
                    break;
                case 'D':
                    args.dim = std::stoi(optarg);
                    if (args.dim != 2 && args.dim != 3) {
                        std::cerr << "Dimension must be 2 or 3" << std::endl;
                        print_nlse_usage(argv[0]);
                        exit(1);
                    }
                    break;
                case 'a':
                    args.anisotropy_file = std::string(optarg);
                    break;
                case 'f':
                    args.focussing_file = std::string(optarg);
                    break;
                case 'L':
                    args.L = std::stod(optarg);
                    break;
                case 'n':
                    args.n = std::stoi(optarg);
                    break;
                case 'T':
                    args.T = std::stod(optarg);
                    break;
                case 't':
                    args.nt = std::stoi(optarg);
                    break;
                case 'S':
                    args.snapshots = std::stoi(optarg);
                    break;
                case 'm':
                    if (std::string(optarg) == "gautschi") {
                        args.method = IntegrationMethod::GAUTSCHI;
                    } else if (std::string(optarg) == "strang") {
                        args.method = IntegrationMethod::STRANG_SPLITTING;
                    } else {
                        std::cerr << "Invalid method: " << optarg << std::endl;
                        print_nlse_usage(argv[0]);
                        exit(1);
                    }
                    break;
                case 'u':
                    args.initial_u_file = std::string(optarg);
                    break;
                case 'o':
                    args.trajectory_file = std::string(optarg);
                    break;
                case 'p': {
                    std::string param_str(optarg);
                    size_t pos = param_str.find('=');
                    if (pos != std::string::npos) {
                        std::string key = param_str.substr(0, pos);
                        double value = std::stod(param_str.substr(pos + 1));
                        args.system_params[key] = value;
                    }
                    break;
                }
                case 'h':
                    print_nlse_usage(argv[0]);
                    exit(0);
                    break;
                default:
                    print_nlse_usage(argv[0]);
                    exit(1);
            }
        }
        
        if (args.initial_u_file.empty()) {
            std::cerr << "Error: initial-u file is required" << std::endl;
            print_nlse_usage(argv[0]);
            exit(1);
        }
        
        if (args.trajectory_file.empty()) {
            std::cerr << "Error: trajectory-file is required" << std::endl;
            print_nlse_usage(argv[0]);
            exit(1);
        }
        
        return args;
    }
    
    static void print_real_wave_usage(const char* program_name) {
        std::cerr << "Usage: " << program_name << " [OPTIONS]\n"
                  << "Options:\n"
                  << "  --device TYPE             Device type (host, cuda) [default: host]\n"
                  << "  --system-type TYPE        System type (klein-gordon, sine-gordon, double-sine-gordon, hyperbolic-sine-gordon, phi4) [default: klein-gordon]\n"
                  << "  --dim N                   Dimension (2 or 3) [default: 2]\n"
                  << "  --anisotropy FILE         Anisotropy coefficient file (c(x) in npy format)\n"
                  << "  --focussing FILE          Focussing coefficient file (m(x) in npy format)\n"
                  << "  --L VALUE                 Domain half-length [default: 10.0]\n"
                  << "  --n VALUE                 Number of grid points per dimension [default: 200]\n"
                  << "  --T VALUE                 Total simulation time [default: 10.0]\n"
                  << "  --nt VALUE                Number of time steps [default: 1000]\n"
                  << "  --snapshots VALUE         Number of snapshots to save [default: 32]\n"
                  << "  --method TYPE             Integration method (gautschi, strang) [default: gautschi]\n"
                  << "  --initial-u FILE          Initial condition file (npy format)\n"
                  << "  --initial-v FILE          Initial velocity file (npy format)\n"
                  << "  --trajectory-file FILE    Output trajectory file (npy format)\n"
                  << "  --velocity-file FILE      Output velocity file (npy format)\n"
                  << "  --param KEY=VALUE         Additional system-specific parameters\n"
                  << "  --help                    Display this help message\n";
    }
    
    static void print_nlse_usage(const char* program_name) {
        std::cerr << "Usage: " << program_name << " [OPTIONS]\n"
                  << "Options:\n"
                  << "  --device TYPE             Device type (host, cuda) [default: host]\n"
                  << "  --system-type TYPE        System type (cubic, cubic-quintic, saturable) [default: cubic]\n"
                  << "  --dim N                   Dimension (2 or 3) [default: 2]\n"
                  << "  --anisotropy FILE         Anisotropy coefficient file (c(x) in npy format)\n"
                  << "  --focussing FILE          Focussing coefficient file (m(x) in npy format)\n"
                  << "  --L VALUE                 Domain half-length [default: 10.0]\n"
                  << "  --n VALUE                 Number of grid points per dimension [default: 200]\n"
                  << "  --T VALUE                 Total simulation time [default: 10.0]\n"
                  << "  --nt VALUE                Number of time steps [default: 1000]\n"
                  << "  --snapshots VALUE         Number of snapshots to save [default: 32]\n"
                  << "  --method TYPE             Integration method (gautschi, strang) [default: gautschi]\n"
                  << "  --initial-u FILE          Initial condition file (npy format)\n"
                  << "  --trajectory-file FILE    Output trajectory file (npy format)\n"
                  << "  --param KEY=VALUE         Additional system-specific parameters\n"
                  << "  --help                    Display this help message\n";
    }
};

#endif
