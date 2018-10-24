cmake_minimum_required(VERSION 3.10)
project(Exercise1)

set(CMAKE_CXX_STANDARD 11)

set(LIBS "${CMAKE_SOURCE_DIR}/libs") #Local Setting

#Compiler options
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")

#External libraries
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lfreeimage")

#Search path for include files
include_directories(${LIBS}/Eigen)

add_executable(Exercise1 FreeImageHelper.cpp main.cpp)

if (UNIX)
    target_link_libraries(Exercise1 freeimage)
endif (UNIX)