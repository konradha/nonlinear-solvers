add_executable(nlwave_bin_complex nlwave_bin_complex.cpp)
target_link_libraries(nlwave_bin_complex PRIVATE host_solvers)
install(TARGETS nlwave_bin_complex RUNTIME DESTINATION bin)

add_executable(nlwave_bin_real nlwave_bin_real.cpp)
target_link_libraries(nlwave_bin_real PRIVATE host_solvers)
install(TARGETS nlwave_bin_real RUNTIME DESTINATION bin)

if(ENABLE_GPU)
    set_source_files_properties(nlwave_bin_complex.cpp PROPERTIES LANGUAGE CUDA)
    set_source_files_properties(nlwave_bin_real.cpp PROPERTIES LANGUAGE CUDA)
    
    target_link_libraries(nlwave_bin_complex PRIVATE device_solvers ${CUDA_LIBRARIES})
    target_link_libraries(nlwave_bin_real PRIVATE device_solvers ${CUDA_LIBRARIES})
    
    set_target_properties(nlwave_bin_complex PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    
    set_target_properties(nlwave_bin_real PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
