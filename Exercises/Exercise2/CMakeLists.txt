cmake_minimum_required(VERSION 3.10)
project(Exercise2)

set(CMAKE_CXX_STANDARD 11)

set(LIBS "${CMAKE_SOURCE_DIR}/libs") #Local Setting

#Compiler options
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -O3")

#Search path for include files
include_directories(${LIBS}/Eigen)

add_executable(Exercise2
    Volume.cpp
    main.cpp
    ImplicitSurface.h
    MarchingCubes.h
    SimpleMesh.h
    Volume.h
)