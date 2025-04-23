# cmake .. -DENABLE_GPU=ON -DCMAKE_CXX_COMPILER=nvcc
cmake_minimum_required(VERSION 3.14)
project(nlsolvers VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(ENABLE_GPU "Enable GPU support" OFF)
option(OPTIMIZE_SIZE "Optimize binaries for size" OFF)
option(OPTIMIZE_SPEED "Optimize for maximum perf" ON)

option(BUILD_NLSE_CUBIC "Build NLSE with cubic nonlinearity" ON)
option(BUILD_NLSE_CUBIC_QUINTIC "Build NLSE with cubic-quintic nonlinearity" ON) 
option(BUILD_NLSE_SATURATING "Build NLSE with saturating nonlinearity" ON)
option(BUILD_SG_SINGLE "Build sGE with a single term" ON)
option(BUILD_SG_DOUBLE "Build sGE with sin(u) + sin(u/2)" ON)
option(BUILD_SG_HYPERBOLIC "Build sGE with sinh(u)" ON)
option(BUILD_KG "Build KGE" ON)
option(BUILD_PHI4 "Build phi-4 model" ON)

option(BUILD_LEGACY_DRIVERS "Build legacy driver binaries" OFF)
option(BUILD_UNIFIED_DRIVERS "Build unified driver binaries" ON)

if(ENABLE_GPU)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86" CACHE STRING "CUDA architectures") 
  set(CMAKE_CXX_COMPILER ${CMAKE_CUDA_COMPILER})
  find_package(CUDAToolkit REQUIRED)

  
  if(OPTIMIZE_SPEED)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2 --use_fast_math --expt-relaxed-constexpr")
  endif()
  include_directories(
    ${CMAKE_SOURCE_DIR}/device/include
  )

  set(CUDA_LIBRARIES 
      CUDA::cusolver
      CUDA::cusparse
      CUDA::cublas
  )

else()
  if(OPTIMIZE_SPEED)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
      add_compile_options(-O2 -march=native -ftree-vectorize -ffast-math)
    endif()
  elseif(OPTIMIZE_SIZE)
    add_compile_options(-Os)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
      add_compile_options(-ffunction-sections -fdata-sections)
      add_link_options(-Wl,--gc-sections)
    endif()
  endif()
  add_definitions(-DEIGEN_NO_DEBUG)
  add_definitions(-DEIGEN_MAX_ALIGN_BYTES=64)
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

if(ENABLE_GPU)
  add_subdirectory(device)
endif()

if(BUILD_UNIFIED_DRIVERS)
  add_subdirectory(drivers)
endif()
