cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(cudaPcl)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)

# boost
set(Boost_USE_STATIC_LIBS   OFF)
find_package (Boost COMPONENTS program_options REQUIRED)

# Eigen
find_package(Eigen3 REQUIRED)

# Adapt it accordingly
SET(CUDA_NVCC_FLAGS  "-arch=sm_35" "--use_fast_math" "-O3" "-lineinfo")

## Include / Link stuff
include_directories(
            ${EIGEN_INCLUDE_DIR}
            ${OPENCV_INCLUDE_DIRS}
            ${CUDA_INCLUDE_DIRS}
            ${PCL_INCLUDE_DIRS}
            ${Boost_INCLUDE_DIR}
            ${PROJECT_SOURCE_DIR}/include
            )

link_directories(
  ${OPENCV_LIBRARY_DIRS}
  ${PCL_LIBRARY_DIRS}
  ${Boost_LIBRARY_DIRS}
)

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

## Library for computing normals
cuda_add_library(cudaPcl SHARED
                    cuda/convolutionSeparable.cu
                    cuda/cuda_pcl_helpers.cu
                    cuda/guidedFilter.cu)

## Test executable
cuda_add_executable(openniSmoothNormals
                        test/openniSmoothNormals.cpp)
target_link_libraries(openniSmoothNormals cudaPcl ${Boost_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})
