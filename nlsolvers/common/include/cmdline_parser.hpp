#ifndef CMDLINE_PARSER_HPP
#define CMDLINE_PARSER_HPP

#include <string>
#include <optional>
#include <stdexcept>
#include <iostream>
#include <cstring>

/**
 * @brief Enumeration of device types
 */
enum class DeviceType {
    HOST,
    CUDA
};

/**
 * @brief Enumeration of system types
 */
enum class SystemType {
    KLEIN_GORDON,
    SINE_GORDON,
    NLSE_CUBIC
};

/**
 * @brief Structure to hold command-line arguments
 */
struct CommandLineArgs {
    DeviceType device = DeviceType::HOST;
    SystemType system_type = SystemType::KLEIN_GORDON;
    int dim = 2;
    double L = 10.0;
    int n = 200;
    double T = 10.0;
    int nt = 1000;
    int snapshots = 32;
    std::string method = "gautschi";
    std::string initial_u_file;
    std::optional<std::string> initial_v_file;
    std::string trajectory_file;
    std::optional<std::string> velocity_file;
    std::optional<std::string> anisotropy_file;
    std::optional<std::string> focussing_file;
};

/**
 * @brief Class for parsing command-line arguments
 */
class CommandLineParser {
public:
    /**
     * @brief Parse command-line arguments for real-valued wave solvers
     * 
     * @param argc Number of arguments
     * @param argv Array of arguments
     * @return CommandLineArgs Parsed arguments
     */
    static CommandLineArgs parse_real_wave_args(int argc, char** argv) {
        CommandLineArgs args;
        
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--device") == 0) {
                if (i + 1 < argc) {
                    if (strcmp(argv[i + 1], "host") == 0) {
                        args.device = DeviceType::HOST;
                    } else if (strcmp(argv[i + 1], "cuda") == 0) {
                        args.device = DeviceType::CUDA;
                    } else {
                        throw std::runtime_error("Invalid device type: " + std::string(argv[i + 1]));
                    }
                    i++;
                } else {
                    throw std::runtime_error("Missing value for --device");
                }
            } else if (strcmp(argv[i], "--system-type") == 0) {
                if (i + 1 < argc) {
                    if (strcmp(argv[i + 1], "klein-gordon") == 0) {
                        args.system_type = SystemType::KLEIN_GORDON;
                    } else if (strcmp(argv[i + 1], "sine-gordon") == 0) {
                        args.system_type = SystemType::SINE_GORDON;
                    } else {
                        throw std::runtime_error("Invalid system type: " + std::string(argv[i + 1]));
                    }
                    i++;
                } else {
                    throw std::runtime_error("Missing value for --system-type");
                }
            } else if (strcmp(argv[i], "--dim") == 0) {
                if (i + 1 < argc) {
                    args.dim = std::stoi(argv[i + 1]);
                    if (args.dim != 2 && args.dim != 3) {
                        throw std::runtime_error("Invalid dimension: " + std::string(argv[i + 1]));
                    }
                    i++;
                } else {
                    throw std::runtime_error("Missing value for --dim");
                }
            } else if (strcmp(argv[i], "--anisotropy") == 0) {
                if (i + 1 < argc) {
                    args.anisotropy_file = argv[i + 1];
                    i++;
                } else {
                    throw std::runtime_error("Missing value for --anisotropy");
                }
            } else if (strcmp(argv[i], "--focussing") == 0) {
                if (i + 1 < argc) {
                    args.focussing_file = argv[i + 1];
                    i++;
                } else {
                    throw std::runtime_error("Missing value for --focussing");
                }
            } else if (strcmp(argv[i], "--L") == 0) {
                if (i + 1 < argc) {
                    args.L = std::stod(argv[i + 1]);
                    i++;
                } else {
                    throw std::runtime_error("Missing value for --L");
                }
            } else if (strcmp(argv[i], "--n") == 0) {
                if (i + 1 < argc) {
                    args.n = std::stoi(argv[i + 1]);
                    i++;
                } else {
                    throw std::runtime_error("Missing value for --n");
                }
            } else if (strcmp(argv[i], "--T") == 0) {
                if (i + 1 < argc) {
                    args.T = std::stod(argv[i + 1]);
                    i++;
                } else {
                    throw std::runtime_error("Missing value for --T");
                }
            } else if (strcmp(argv[i], "--nt") == 0) {
                if (i + 1 < argc) {
                    args.nt = std::stoi(argv[i + 1]);
                    i++;
                } else {
                    throw std::runtime_error("Missing value for --nt");
                }
            } else if (strcmp(argv[i], "--snapshots") == 0) {
                if (i + 1 < argc) {
                    args.snapshots = std::stoi(argv[i + 1]);
                    i++;
                } else {
                    throw std::runtime_error("Missing value for --snapshots");
                }
            } else if (strcmp(argv[i], "--method") == 0) {
                if (i + 1 < argc) {
                    args.method = argv[i + 1];
                    i++;
                } else {
                    throw std::runtime_error("Missing value for --method");
                }
            } else if (strcmp(argv[i], "--initial-u") == 0) {
                if (i + 1 < argc) {
                    args.initial_u_file = argv[i + 1];
                    i++;
                } else {
                    throw std::runtime_error("Missing value for --initial-u");
                }
            } else if (strcmp(argv[i], "--initial-v") == 0) {
                if (i + 1 < argc) {
                    args.initial_v_file = argv[i + 1];
                    i++;
                } else {
                    throw std::runtime_error("Missing value for --initial-v");
                }
            } else if (strcmp(argv[i], "--trajectory-file") == 0) {
                if (i + 1 < argc) {
                    args.trajectory_file = argv[i + 1];
                    i++;
                } else {
                    throw std::runtime_error("Missing value for --trajectory-file");
                }
            } else if (strcmp(argv[i], "--velocity-file") == 0) {
                if (i + 1 < argc) {
                    args.velocity_file = argv[i + 1];
                    i++;
                } else {
                    throw std::runtime_error("Missing value for --velocity-file");
                }
            } else {
                throw std::runtime_error("Unknown argument: " + std::string(argv[i]));
            }
        }
        
        // Check required arguments
        if (args.initial_u_file.empty()) {
            throw std::runtime_error("Missing required argument: --initial-u");
        }
        if (args.trajectory_file.empty()) {
            throw std::runtime_error("Missing required argument: --trajectory-file");
        }
        
        return args;
    }
    
    /**
     * @brief Parse command-line arguments for NLSE solvers
     * 
     * @param argc Number of arguments
     * @param argv Array of arguments
     * @return CommandLineArgs Parsed arguments
     */
    static CommandLineArgs parse_nlse_args(int argc, char** argv) {
        CommandLineArgs args;
        
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--device") == 0) {
                if (i + 1 < argc) {
                    if (strcmp(argv[i + 1], "host") == 0) {
                        args.device = DeviceType::HOST;
                    } else if (strcmp(argv[i + 1], "cuda") == 0) {
                        args.device = DeviceType::CUDA;
                    } else {
                        throw std::runtime_error("Invalid device type: " + std::string(argv[i + 1]));
                    }
                    i++;
                } else {
                    throw std::runtime_error("Missing value for --device");
                }
            } else if (strcmp(argv[i], "--system-type") == 0) {
                if (i + 1 < argc) {
                    if (strcmp(argv[i + 1], "nlse-cubic") == 0) {
                        args.system_type = SystemType::NLSE_CUBIC;
                    } else {
                        throw std::runtime_error("Invalid system type: " + std::string(argv[i + 1]));
                    }
                    i++;
                } else {
                    throw std::runtime_error("Missing value for --system-type");
                }
            } else if (strcmp(argv[i], "--dim") == 0) {
                if (i + 1 < argc) {
                    args.dim = std::stoi(argv[i + 1]);
                    if (args.dim != 2 && args.dim != 3) {
                        throw std::runtime_error("Invalid dimension: " + std::string(argv[i + 1]));
                    }
                    i++;
                } else {
                    throw std::runtime_error("Missing value for --dim");
                }
            } else if (strcmp(argv[i], "--anisotropy") == 0) {
                if (i + 1 < argc) {
                    args.anisotropy_file = argv[i + 1];
                    i++;
                } else {
                    throw std::runtime_error("Missing value for --anisotropy");
                }
            } else if (strcmp(argv[i], "--focussing") == 0) {
                if (i + 1 < argc) {
                    args.focussing_file = argv[i + 1];
                    i++;
                } else {
                    throw std::runtime_error("Missing value for --focussing");
                }
            } else if (strcmp(argv[i], "--L") == 0) {
                if (i + 1 < argc) {
                    args.L = std::stod(argv[i + 1]);
                    i++;
                } else {
                    throw std::runtime_error("Missing value for --L");
                }
            } else if (strcmp(argv[i], "--n") == 0) {
                if (i + 1 < argc) {
                    args.n = std::stoi(argv[i + 1]);
                    i++;
                } else {
                    throw std::runtime_error("Missing value for --n");
                }
            } else if (strcmp(argv[i], "--T") == 0) {
                if (i + 1 < argc) {
                    args.T = std::stod(argv[i + 1]);
                    i++;
                } else {
                    throw std::runtime_error("Missing value for --T");
                }
            } else if (strcmp(argv[i], "--nt") == 0) {
                if (i + 1 < argc) {
                    args.nt = std::stoi(argv[i + 1]);
                    i++;
                } else {
                    throw std::runtime_error("Missing value for --nt");
                }
            } else if (strcmp(argv[i], "--snapshots") == 0) {
                if (i + 1 < argc) {
                    args.snapshots = std::stoi(argv[i + 1]);
                    i++;
                } else {
                    throw std::runtime_error("Missing value for --snapshots");
                }
            } else if (strcmp(argv[i], "--method") == 0) {
                if (i + 1 < argc) {
                    args.method = argv[i + 1];
                    i++;
                } else {
                    throw std::runtime_error("Missing value for --method");
                }
            } else if (strcmp(argv[i], "--initial-u") == 0) {
                if (i + 1 < argc) {
                    args.initial_u_file = argv[i + 1];
                    i++;
                } else {
                    throw std::runtime_error("Missing value for --initial-u");
                }
            } else if (strcmp(argv[i], "--trajectory-file") == 0) {
                if (i + 1 < argc) {
                    args.trajectory_file = argv[i + 1];
                    i++;
                } else {
                    throw std::runtime_error("Missing value for --trajectory-file");
                }
            } else {
                throw std::runtime_error("Unknown argument: " + std::string(argv[i]));
            }
        }
        
        // Check required arguments
        if (args.initial_u_file.empty()) {
            throw std::runtime_error("Missing required argument: --initial-u");
        }
        if (args.trajectory_file.empty()) {
            throw std::runtime_error("Missing required argument: --trajectory-file");
        }
        
        return args;
    }
};

#endif // CMDLINE_PARSER_HPP
