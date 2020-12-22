cmake_minimum_required(VERSION 2.8)

set(PROJECT_NAME "Optimisation")

project( ${PROJECT_NAME} )


set(CMAKE_CXX_FLAGS "-std=c++11")

find_package( OpenCV REQUIRED)

message("CV version: " ${OpenCV_VERSION})
message("CV libs : " ${OpenCV_LIBS})
# message("optflow: " ${OpenCV_CUDAOPTFLOW_FOUND})
# message("include dirs: " ${OpenCV_INCLUDE_DIRS})
# message("install path: " ${OpenCV_INSTALL_PATH})
# message("modules: " ${OpenCV_LIB_COMPONENTS})



find_package(Eigen3 3.2.0 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})


file(GLOB_RECURSE HELPER_FILES ${PROJECT_SOURCE_DIR}/helpers/*)


add_executable( "basic_gna" basic_gna_example.cpp ${HELPER_FILES} )
target_link_libraries( "basic_gna" ${OpenCV_LIBS} )


add_executable( "lma" lma.cpp ${HELPER_FILES} )
target_link_libraries( "lma" ${OpenCV_LIBS} )