#ifndef BOUNDARIES_HPP
#define BOUNDARIES_HPP

#include <Eigen/Dense>


// We assume all vectors u, v to be actual vectors, ie. their shapes to be either
// (nx * ny, 1) or (nx * ny * nz, 1)

template <typename Float>
void neumann_bc(Eigen::VectorX<Float> & u,
                Eigen::VectorX<Float> & v,
                const uint32_t nx, 
                const uint32_t ny,
                const Float dx, 
                const Float dy) {
    // port of the Numpy-y
    /*
     def neumann_bc(self, u, v, i, tau=None,):
        u[0, 1:-1] = u[1, 1:-1]
        u[-1, 1:-1] = u[-2, 1:-1]
        u[:, 0] = u[:, 1]
        u[:, -1] = u[:, -2]

        v[0, 1:-1] = 0
        v[-1, 1:-1] = 0
        v[1:-1, 0] = 0
        v[1:-1, -1] = 0
     */
    Eigen::Map<Eigen::Matrix<Float, Eigen::Dynamic, Eigen::Dynamic>> u_map(u.data(), nx, ny);
    Eigen::Map<Eigen::Matrix<Float, Eigen::Dynamic, Eigen::Dynamic>> v_map(v.data(), nx, ny);
    u_map.row(0).segment(1, ny-2) = u_map.row(1).segment(1, ny-2);
    u_map.row(nx-1).segment(1, ny-2) = u_map.row(nx-2).segment(1, ny-2);
    u_map.col(0) = u_map.col(1);
    u_map.col(ny-1) = u_map.col(ny-2);
    v_map.row(0).segment(1, ny-2).setZero();
    v_map.row(nx-1).segment(1, ny-2).setZero();
    v_map.block(1, 0, nx-2, 1).setZero();
    v_map.block(1, ny-1, nx-2, 1).setZero();
} 

template <typename Float>
void neumann_bc_no_velocity(Eigen::VectorX<Float> & u,
                const uint32_t nx, 
                const uint32_t ny) {
    // port of the Numpy-y
    // Used because we can save a lot of cycles that way.

    /*
     def neumann_bc(self, u, v, i, tau=None,):
        u[0, 1:-1] = u[1, 1:-1]
        u[-1, 1:-1] = u[-2, 1:-1]
        u[:, 0] = u[:, 1]
        u[:, -1] = u[:, -2]
     */
    Eigen::Map<Eigen::Matrix<Float, Eigen::Dynamic, Eigen::Dynamic>> u_map(u.data(), nx, ny);
    u_map.row(0).segment(1, ny-2) = u_map.row(1).segment(1, ny-2);
    u_map.row(nx-1).segment(1, ny-2) = u_map.row(nx-2).segment(1, ny-2);
    u_map.col(0) = u_map.col(1);
    u_map.col(ny-1) = u_map.col(ny-2);
}


#endif
