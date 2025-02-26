cmake_minimum_required(VERSION 3.14)
project(nlsolvers VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(OPTIMIZE_SIZE "Optimize binaries for size" ON)
option(BUILD_NLSE_CUBIC "Build NLSE with cubic nonlinearity" ON)
option(BUILD_NLSE_CUBIC_QUINTIC "Build NLSE with cubic-quintic nonlinearity" ON) 
option(BUILD_NLSE_SATURATING "Build NLSE with saturating nonlinearity" ON)

option(BUILD_SG_SINGLE "Build Sine-Gordon equation (single)" ON)

if(OPTIMIZE_SIZE)
    add_compile_options(-Os)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(-ffunction-sections -fdata-sections)
        add_link_options(-Wl,--gc-sections)
    endif()
endif()

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

set(LIBNPY_DIR "${CMAKE_SOURCE_DIR}/../libnpy" CACHE PATH "Path to libnpy directory")
if(EXISTS "${LIBNPY_DIR}/include/npy.hpp")
    include_directories(${LIBNPY_DIR}/include)
    add_definitions(-DHAVE_LIBNPY)
else()
    message(FATAL_ERROR "libnpy not found at ${LIBNPY_DIR}. Specify with -DLIBNPY_DIR=path/to/libnpy")
endif()

include_directories(
    ${CMAKE_SOURCE_DIR}/common/include
    ${CMAKE_SOURCE_DIR}/host/include
)

add_subdirectory(common)
add_subdirectory(host)
