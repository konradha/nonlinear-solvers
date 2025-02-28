if(CMAKE_CXX_COMPILER_ID MATCHES "NVIDIA")
    include_directories(
        ${CMAKE_SOURCE_DIR}/common/include
        ${CMAKE_SOURCE_DIR}/device/include
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )

    add_library(device_solvers INTERFACE)
    target_include_directories(device_solvers INTERFACE 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    )
    target_link_libraries(device_solvers INTERFACE common)
    add_subdirectory(drivers)
endif()
