# Device headers
set(DEVICE_HEADERS
    include/boundaries.cuh
    include/common_kernels.cuh
    
    # Common headers
    include/common/solver_base_dev.hpp
    include/common/cmdline_parser_dev.hpp
    include/common/solver_factory_dev.hpp
    
    # NLSE headers
    include/nlse/nlse_solver_base_dev.hpp
    include/nlse.cuh
    include/nlse_dev.hpp
    include/nlse_cubic_quintic.cuh
    include/nlse_cubic_quintic_dev.hpp
    include/nlse_saturating.cuh
    include/nlse_saturating_dev.hpp
    
    # Real-space headers
    include/realspace/realspace_solver_base_dev.hpp
    include/kg_single.cuh
    include/kg_dev.hpp
    include/phi4_single.cuh
    include/phi4_dev.hpp
    include/sg_single.cuh
    include/sg_single_dev.hpp
    include/sg_double.cuh
    include/sg_double_dev.hpp
    include/sg_hyperbolic.cuh
    include/sg_hyperbolic_dev.hpp
)

add_library(device_solvers INTERFACE)
target_include_directories(device_solvers INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(device_solvers INTERFACE common)

add_subdirectory(drivers)

install(FILES ${DEVICE_HEADERS}
    DESTINATION include/device
)
