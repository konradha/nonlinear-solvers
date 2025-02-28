if(BUILD_SG_SINGLE)
    add_executable(sg_single_dev sg_single_driver_dev.cpp)
    set_source_files_properties(sg_single_driver_dev.cpp PROPERTIES LANGUAGE CUDA)
    target_link_libraries(sg_single_dev PRIVATE device_solvers)
    set_target_properties(sg_single_dev PROPERTIES 
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    install(TARGETS sg_single_dev RUNTIME DESTINATION bin)
endif()
