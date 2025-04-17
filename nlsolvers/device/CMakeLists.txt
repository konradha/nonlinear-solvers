set(DEVICE_HEADERS
    include/boundaries.cuh
    include/sg_single.cuh
    include/sg_single_dev.hpp

    include/common_kernels.cuh
    include/kg_single_dev.hpp
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
