cmake_minimum_required(VERSION 3.10)
project(Exercise4)

# -> REQUIRED PACKAGES
find_package(Eigen3 3.3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(OpenCV REQUIRED)
include_directories(${OPENCV_INCLUDE_DIR})

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})
# <- 

#set(LIBS "${CMAKE_CURRENT_SOURCE_DIR}/libs/")

#Compile options
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2")
add_definitions(-DHOME_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
add_definitions(-DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/assets/rgbd_dataset_freiburg3_teddy/")

#Search path for include files
#include_directories(${LIBS}/Eigen)

set(SOURCE_FILES_CPP
	FreeImageHelper.cpp
)

set(SOURCE_FILES_H
	World.h
	FreeImageHelper.h
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build)

add_executable(exercise4 main.cpp ${SOURCE_FILES_CPP} ${SOURCE_FILES_H})

target_link_libraries(exercise4 freeimage opencv_core opencv_highgui opencv_features2d opencv_imgcodecs ${CERES_LIBRARIES})