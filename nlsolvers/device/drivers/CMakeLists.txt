# Function to add CUDA driver
function(add_cuda_driver name source)
    add_executable(${name} ${source})
    set_source_files_properties(${source} PROPERTIES LANGUAGE CUDA)
    target_link_libraries(${name} PRIVATE device_solvers ${CUDA_LIBRARIES})
    set_target_properties(${name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    install(TARGETS ${name} RUNTIME DESTINATION bin)
endfunction(add_cuda_driver)

# Unified drivers
if(BUILD_UNIFIED_DRIVERS)
    add_cuda_driver(nlse_driver_dev_unified nlse_driver_dev_unified.cpp)
    add_cuda_driver(realspace_driver_dev_unified realspace_driver_dev_unified.cpp)
endif()

# Legacy drivers - these will be removed once the unified drivers are stable
if(BUILD_LEGACY_DRIVERS)
    if(BUILD_SG_SINGLE)
        add_cuda_driver(sg_single_dev sg_single_driver_dev.cpp)
    endif()

    if(BUILD_SG_DOUBLE)
        add_cuda_driver(sg_double_dev sg_double_driver_dev.cpp)
    endif()

    if(BUILD_SG_HYPERBOLIC)
        add_cuda_driver(sg_hyperbolic_dev sg_hyperbolic_driver_dev.cpp)
    endif()

    if(BUILD_KG)
        add_cuda_driver(kg_dev kg_driver_dev.cpp)
    endif()

    if(BUILD_PHI4)
        add_cuda_driver(phi4_dev phi4_driver_dev.cpp)
    endif()

    if(BUILD_NLSE_CUBIC)
        add_cuda_driver(nlse_dev nlse_driver_dev.cpp)
    endif()

    if(BUILD_NLSE_CUBIC_QUINTIC)
        add_cuda_driver(nlse_cubic_quintic_dev nlse_cubic_quintic_driver_dev.cpp)
    endif()

    if(BUILD_NLSE_SATURATING)
        add_cuda_driver(nlse_saturating_dev nlse_saturating_driver_dev.cpp)
    endif()
endif()
