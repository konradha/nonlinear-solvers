cmake_minimum_required(VERSION 3.10)
project(nlsolvers VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find dependencies
find_package(Eigen3 3.3 REQUIRED)
find_package(cnpy REQUIRED)

# CUDA support
option(WITH_CUDA "Build with CUDA support" OFF)
if(WITH_CUDA)
    enable_language(CUDA)
    find_package(CUDA REQUIRED)
    add_definitions(-DWITH_CUDA)
endif()

# Add subdirectories
add_subdirectory(common)
add_subdirectory(core)

# Install headers
install(DIRECTORY common/include/ DESTINATION include/common)
install(DIRECTORY core/include/ DESTINATION include/core)
