set(HOST_HEADERS 
    include/boundaries.hpp
    include/boundaries_3d.hpp
    # TODO: finish Laplacians (ie. Radiating Sommerfeld) and the boundary applications
    # - du/dn = 0 for no-flux
    # - Radiation application: Laplacian gets characteristic wave speed in beginning
    #   -> simple FFT estimate for c
    # - apply boundaries more precisely to see actual behavior!
    include/eigen_krylov_real.hpp
    include/eigen_krylov_complex.hpp
    include/kg_solver.hpp
    include/nlse_saturating_solver.hpp
    include/sg_hyperbolic_solver.hpp 
    include/nlse_cubic_quintic_solver.hpp
    include/phi4_solver.hpp
    include/sg_single_solver.hpp 
    include/nlse_cubic_solver.hpp
    include/sg_double_solver.hpp

    nlse_cubic_gautschi_solver.hpp # let's try and bring the sEWI integrator to life

    include/sg_single_sv_solver.hpp
)

add_library(host_solvers INTERFACE)
target_include_directories(host_solvers INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(host_solvers INTERFACE common)

install(FILES ${HOST_HEADERS}
    DESTINATION include/host
)

add_subdirectory(drivers)
