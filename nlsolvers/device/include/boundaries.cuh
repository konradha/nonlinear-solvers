#ifndef BOUNDARIES_CUH
#define BOUNDARIES_CUH

#include <cuda_runtime.h>
#include <cuda.h>

// benchmarked to be fastest for small (nx, ny <= 5000) grid sizes; after that
// considerable overhead and we shoudl think about how to better employ warps
template <typename Float>
void neumann_bc_no_velocity_blocking(Float* u_dev, uint32_t nx, uint32_t ny) {
    cudaMemcpy(
        u_dev + 0 * ny + 1,      
        u_dev + 1 * ny + 1,       
        (ny - 2) * sizeof(Float),  
        cudaMemcpyDeviceToDevice
    ); 
    cudaMemcpy(
        u_dev + (nx - 1) * ny + 1, 
        u_dev + (nx - 2) * ny + 1, 
        (ny - 2) * sizeof(Float),
        cudaMemcpyDeviceToDevice
    ); 
    cudaMemcpy2D(
        u_dev + 0,             
        ny * sizeof(Float),   
        u_dev + 1,              
        ny * sizeof(Float),     
        sizeof(Float),          
        nx,                     
        cudaMemcpyDeviceToDevice
    ); 
    cudaMemcpy2D(
        u_dev + (ny - 1),     
        ny * sizeof(Float),
        u_dev + (ny - 2),       
        ny * sizeof(Float),
        sizeof(Float),
        nx,
        cudaMemcpyDeviceToDevice
    );
}

#endif
