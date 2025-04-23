add_library(core_common INTERFACE)
target_include_directories(core_common INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(core_common INTERFACE common)

# Unified binaries
add_executable(nlwave_bin_real drivers/nlwave_bin_real.cpp)
target_link_libraries(nlwave_bin_real PRIVATE core_common)

add_executable(nlwave_bin_nlse drivers/nlwave_bin_nlse.cpp)
target_link_libraries(nlwave_bin_nlse PRIVATE core_common)

# Install binaries
install(TARGETS nlwave_bin_real nlwave_bin_nlse
    RUNTIME DESTINATION bin
)
