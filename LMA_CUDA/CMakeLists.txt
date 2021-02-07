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


add_subdirectory(src)


# find_package(Eigen3 3.3.9 REQUIRED PATHS "/home/hoqe4365/Documents/libs/eigen-3.3.9/")
set(EIGEN3_INCLUDE_DIR "/home/hoqe4365/Documents/libs/eigen-3.3.9/")

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(include)


message("Eigen include dirs: " ${EIGEN3_INCLUDE_DIR})

# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)




file(GLOB_RECURSE HELPER_FILES ${PROJECT_SOURCE_DIR}/helpers/*)


set (TGT_NAME "lma_with_cuda")
# add_executable( ${TGT_NAME} ${TGT_NAME}.cpp ${HELPER_FILES} )
cuda_add_executable( ${TGT_NAME} ${TGT_NAME}.cpp ${HELPER_FILES} )
target_link_libraries( ${TGT_NAME} ${OpenCV_LIBS} cusparse_cholesky_solver)

