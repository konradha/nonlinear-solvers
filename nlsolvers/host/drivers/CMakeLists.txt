# # TODO:
# # [ ] all Gautschi-type integrators (sEWI, NLSE)
# # [ ] all SV-type integrators (KGE, SGE ...)
# # [ ] comparison runners: distance, energies, for all phenomena to check
# 
# if(BUILD_NLSE_CUBIC)
#     if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/nlse_cubic_driver.cpp")
#         add_executable(nlse_cubic nlse_cubic_driver.cpp)
#         target_link_libraries(nlse_cubic PRIVATE host_solvers)
#         install(TARGETS nlse_cubic RUNTIME DESTINATION bin)
#     endif()
# 
#     if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/nlse_cubic_gautschi_driver.cpp")
#         add_executable(nlse_cubic_gautschi nlse_cubic_gautschi_driver.cpp)
#         target_link_libraries(nlse_cubic_gautschi PRIVATE host_solvers)
#         install(TARGETS nlse_cubic_gautschi RUNTIME DESTINATION bin)
#     endif()
# 
# endif()
# 
# if(BUILD_NLSE_CUBIC_QUINTIC)
#     if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/nlse_cubic_quintic_driver.cpp")
#         add_executable(nlse_cubic_quintic nlse_cubic_quintic_driver.cpp)
#         target_link_libraries(nlse_cubic_quintic PRIVATE host_solvers)
#         install(TARGETS nlse_cubic_quintic RUNTIME DESTINATION bin)
#     endif()
#     
#     if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/nlse_cubic_quintic_gautschi_driver.cpp")
#         add_executable(nlse_cubic_quintic_gautschi nlse_cubic_quintic_gautschi_driver.cpp)
#         target_link_libraries(nlse_cubic_quintic_gautschi PRIVATE host_solvers)
#         install(TARGETS nlse_cubic_quintic_gautschi RUNTIME DESTINATION bin)
#     endif()
# endif()
# 
# if(BUILD_NLSE_SATURATING)
#     if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/nlse_saturating_driver.cpp")
#         add_executable(nlse_saturating nlse_saturating_driver.cpp)
#         target_link_libraries(nlse_saturating PRIVATE host_solvers)
#         install(TARGETS nlse_saturating RUNTIME DESTINATION bin)
#     endif()
# endif()
# 
# if(BUILD_SG_SINGLE)
#     if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/sg_single_driver.cpp")
#         add_executable(sg_single sg_single_driver.cpp)
#         target_link_libraries(sg_single PRIVATE host_solvers)
#         install(TARGETS sg_single RUNTIME DESTINATION bin)
#     endif()
# 
#     if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/sg_single_driver_sv.cpp")
#         add_executable(sg_single_sv sg_single_driver_sv.cpp)
#         target_link_libraries(sg_single_sv PRIVATE host_solvers)
#         install(TARGETS sg_single_sv RUNTIME DESTINATION bin)
#     endif()
# endif()
# 
# if(BUILD_SG_DOUBLE)
#     if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/sg_double_driver.cpp")
#         add_executable(sg_double sg_double_driver.cpp)
#         target_link_libraries(sg_double PRIVATE host_solvers)
#         install(TARGETS sg_double RUNTIME DESTINATION bin)
#     endif()
# endif()
# 
# if(BUILD_SG_HYPERBOLIC)
#     if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/sg_hyperbolic_driver.cpp")
#         add_executable(sg_hyperbolic sg_hyperbolic_driver.cpp)
#         target_link_libraries(sg_hyperbolic PRIVATE host_solvers)
#         install(TARGETS sg_hyperbolic RUNTIME DESTINATION bin)
#     endif()
# endif()
# 
# if(BUILD_KG)
#     if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/kg_driver.cpp")
#         add_executable(kg kg_driver.cpp)
#         target_link_libraries(kg PRIVATE host_solvers)
#         install(TARGETS kg RUNTIME DESTINATION bin)
#     endif()
# endif()
# 
# if(BUILD_PHI4)
#     if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/phi4_driver.cpp")
#         add_executable(phi4 phi4_driver.cpp)
#         target_link_libraries(phi4 PRIVATE host_solvers)
#         install(TARGETS phi4 RUNTIME DESTINATION bin)
#     endif()
# endif()
# 
# add_library(project_common_deps INTERFACE)
# target_link_libraries(project_common_deps INTERFACE rt dl)
# 
# ###### 3D DRIVERS
# if(BUILD_NLSE_CUBIC)
#     ## SS2
#     if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/nlse_cubic_driver_3d.cpp")
#         add_executable(nlse_cubic_host_3d nlse_cubic_driver_3d.cpp)
#         target_link_libraries(nlse_cubic_host_3d PRIVATE project_common_deps)
#         target_link_libraries(nlse_cubic_host_3d PRIVATE host_solvers)
# 	target_link_libraries(nlse_cubic_host_3d PRIVATE ${MATH_LIBRARIES})	
#         install(TARGETS nlse_cubic_host_3d RUNTIME DESTINATION bin)
#     endif()
#     if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/nlse_cubic_gautschi_driver_3d.cpp")
#         add_executable(nlse_cubic_host_gautschi_3d nlse_cubic_gautschi_driver_3d.cpp)
#         target_link_libraries(nlse_cubic_host_gautschi_3d PRIVATE project_common_deps)
#         target_link_libraries(nlse_cubic_host_gautschi_3d PRIVATE host_solvers)
# 	target_link_libraries(nlse_cubic_host_gautschi_3d PRIVATE ${MATH_LIBRARIES})
#         install(TARGETS nlse_cubic_host_gautschi_3d RUNTIME DESTINATION bin)
#     endif()
# endif()
# 
# if(BUILD_KG)
#     ## Gautschi
#     if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/kg_driver_3d.cpp")
#         add_executable(kg_driver_3d kg_driver_3d.cpp)
#         target_link_libraries(kg_driver_3d PRIVATE host_solvers)
#         install(TARGETS kg_driver_3d RUNTIME DESTINATION bin)
#     endif()
#     ## Stormer-Verlet (SpMV)
#     if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/kg_sv_driver_3d.cpp")
#         add_executable(kg_sv_driver_3d kg_sv_driver_3d.cpp)
#         target_link_libraries(kg_sv_driver_3d PRIVATE host_solvers)
#         install(TARGETS kg_sv_driver_3d RUNTIME DESTINATION bin)
#     endif()
# endif()
# 
# 
# add_executable(scipy_test_host test_scipy_matfunc.cpp)
# target_link_libraries(scipy_test_host PRIVATE host_solvers)
# install(TARGETS scipy_test_host RUNTIME DESTINATION bin)
