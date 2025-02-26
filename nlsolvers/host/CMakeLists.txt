set(HOST_HEADERS
    include/eigen_krylov_complex.hpp
    include/eigen_krylov_real.hpp
    include/nlse_cubic_solver.hpp
    include/nlse_cubic_quintic_solver.hpp
    include/nlse_saturating_solver.hpp
    include/sg_single_solver.hpp
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
