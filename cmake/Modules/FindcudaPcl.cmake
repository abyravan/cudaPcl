# - This module tries to find the package "cudaPcl" and its components
# Assumes that env variable: CUDAPCL_ROOT_DIR is set
# Once done this will define
#  cudaPcl_FOUND 		 - System has cudaPcl
#  cudaPcl_INCLUDE_DIRS - The cudaPcl include directories
#  cudaPcl_LIBRARY_DIRS - The directories containing the libraries needed to use cudaPcl
#  cudaPcl_LIBRARIES    - The libraries needed to use cudaPcl
#
# Written by Arunkumar Byravan
set(CUDAPCL_INCLUDE_DIR $ENV{CUDAPCL_ROOT_DIR}/include)

find_path(CUDAPCL_LIBRARY_DIR libcudaPcl.so
    $ENV{CUDAPCL_ROOT_DIR}/lib
)

if(CUDAPCL_LIBRARY_DIR)
    set(cudaPcl_INCLUDE_DIRS ${cudaPcl_INCLUDE_DIRS} ${CUDAPCL_INCLUDE_DIR})
    set(cudaPcl_LIBRARY_DIRS ${cudaPcl_LIBRARY_DIRS} ${CUDAPCL_LIBRARY_DIR})
    set(cudaPcl_LIBRARIES ${cudaPcl_LIBRARIES} ${CUDAPCL_LIBRARY_DIR}/libcudaPcl.so)
endif()

#message(STATUS "cudaPcl_INCLUDE_DIRS: ${cudaPcl_INCLUDE_DIRS}")
#message(STATUS "cudaPcl_LIBRARY_DIRS: ${cudaPcl_LIBRARY_DIRS}")
#message(STATUS "cudaPcl_LIBRARIES: ${cudaPcl_LIBRARIES}")

# handle the QUIETLY and REQUIRED arguments and set cudaPcl_FOUND to TRUE if 
# all listed variables are TRUE
INCLUDE(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(cudaPcl DEFAULT_MSG cudaPcl_INCLUDE_DIRS cudaPcl_LIBRARIES)

IF (cudaPcl_FOUND)
   INCLUDE(CheckLibraryExists)
   CHECK_LIBRARY_EXISTS(${cudaPcl_LIBRARIES})
ENDIF (cudaPcl_FOUND)

MARK_AS_ADVANCED(cudaPcl_INCLUDE_DIRS cudaPcl_LIBRARY_DIRS cudaPcl_LIBRARIES)
