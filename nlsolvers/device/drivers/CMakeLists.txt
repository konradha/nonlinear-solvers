function(add_cuda_driver name source)
    add_executable(${name} ${source})
    set_source_files_properties(${source} PROPERTIES LANGUAGE CUDA)
    target_link_libraries(${name} PRIVATE device_solvers ${CUDA_LIBRARIES})
    set_target_properties(${name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    install(TARGETS ${name} RUNTIME DESTINATION bin)
endfunction()

if(BUILD_SG_SINGLE)
    add_cuda_driver(sg_single_dev sg_single_driver_dev.cpp)
endif()
