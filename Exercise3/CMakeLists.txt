cmake_minimum_required(VERSION 3.10)
project(Exercise3)

set(CMAKE_CXX_STANDARD 14)

# Set variables
set(LIBS "${CMAKE_SOURCE_DIR}/libs") #Local Setting

# Compiler options
set(CMAKE_CXX_FLAGS "-std=c++14 ${CMAKE_CXX_FLAGS} -w -O3")
add_definitions(-DPROJECT_DIR="${PROJECT_SOURCE_DIR}")

# Eigen Library
include_directories(${LIBS}/Eigen)

# Flann Library
include_directories(${LIBS}/Flann)

# Set files to be compiled
set(HEADER_FILES
        Eigen.h
        SimpleMesh.h
        PointCloud.h
        NearestNeighbor.h
        ProcrustesAligner.h
        ICPOptimizer.h
        )

add_executable(Exercise3 main.cpp ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(Exercise3 ${FLANN_LIBRARIES})