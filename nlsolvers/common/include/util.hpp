#ifndef UTIL_HPP
#define UTIL_HPP

#include <cnpy.h>
#include <Eigen/Dense>
#include <string>
#include <vector>
#include <complex>
#include <stdexcept>

/**
 * @brief Read data from a NumPy file
 * 
 * @tparam T Data type (double or std::complex<double>)
 * @param filename Path to the NumPy file
 * @param shape Output parameter for the shape of the data
 * @return Eigen::VectorX<T> Vector containing the data
 */
template <typename T>
Eigen::VectorX<T> read_from_npy(const std::string& filename, std::vector<uint32_t>& shape) {
    try {
        cnpy::NpyArray arr = cnpy::npy_load(filename);
        shape = arr.shape;
        
        // Calculate total size
        size_t total_size = 1;
        for (auto dim : shape) {
            total_size *= dim;
        }
        
        // Create Eigen vector
        Eigen::VectorX<T> result(total_size);
        
        // Copy data
        if constexpr (std::is_same_v<T, double>) {
            // For double
            double* data = arr.data<double>();
            for (size_t i = 0; i < total_size; ++i) {
                result(i) = data[i];
            }
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            // For complex<double>
            std::complex<double>* data = arr.data<std::complex<double>>();
            for (size_t i = 0; i < total_size; ++i) {
                result(i) = data[i];
            }
        }
        
        return result;
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to read from NumPy file: " + std::string(e.what()));
    }
}

/**
 * @brief Save data to a NumPy file
 * 
 * @tparam T Data type (double or std::complex<double>)
 * @param filename Path to the NumPy file
 * @param data Vector containing the data
 * @param shape Shape of the data
 */
template <typename T>
void save_to_npy(const std::string& filename, const Eigen::VectorX<T>& data, const std::vector<uint32_t>& shape) {
    try {
        // Calculate total size
        size_t total_size = 1;
        for (auto dim : shape) {
            total_size *= dim;
        }
        
        // Check size
        if (data.size() != total_size) {
            throw std::runtime_error("Data size does not match shape");
        }
        
        // Save data
        if constexpr (std::is_same_v<T, double>) {
            // For double
            std::vector<double> data_vec(total_size);
            for (size_t i = 0; i < total_size; ++i) {
                data_vec[i] = data(i);
            }
            cnpy::npy_save(filename, data_vec.data(), shape);
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            // For complex<double>
            std::vector<std::complex<double>> data_vec(total_size);
            for (size_t i = 0; i < total_size; ++i) {
                data_vec[i] = data(i);
            }
            cnpy::npy_save(filename, data_vec.data(), shape);
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to save to NumPy file: " + std::string(e.what()));
    }
}

#endif // UTIL_HPP
