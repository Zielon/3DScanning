cmake_minimum_required(VERSION 3.10)
project(tracker-lib VERSION 1.0.0 DESCRIPTION "A library for tracking")

# Set variables
set(LIBS "${CMAKE_SOURCE_DIR}/external-libs") #Local Setting

# Compiler settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -O3")

# Build directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/tracker-lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/tracker-lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/tracker-test)

# ---------> REQUIRED PACKAGES

if (UNIX)
    find_package(OpenCV REQUIRED)
    include_directories(${OPENCV_INCLUDE_DIR})

    find_package(Ceres REQUIRED)
    include_directories(${CERES_INCLUDE_DIRS})
    include_directories(include ${PROJECT_BINARY_DIR})
endif (UNIX)

# Eigen Library
include_directories(${LIBS}/Eigen)

# Flann Library
include_directories(${LIBS}/Flann)


# ------------Compile dll library (deploy)
add_library(
        tracker-lib SHARED
        reconstruction/source/Tracker.cpp
        reconstruction/source/ICP.cpp
        dataStream/source/DatasetVideoStreamReader.cpp
        ExportDLL.cpp
        reconstruction/source/NearestNeighbor.hpp
        Eigen.h
)

#--------------Compile library (Linux and Apple development)
if (UNIX)
    target_link_libraries(
            tracker-lib
            freeimage
            opencv_core
            opencv_highgui
            opencv_features2d
            opencv_imgcodecs
            freeimage
            ${CERES_LIBRARIES}
            ${FLANN_LIBRARIES}
    )
endif (UNIX)

target_link_libraries(
        tracker-lib
        ${FLANN_LIBRARIES}
)

add_executable(
        tracker-test
        main.cpp
        tests/windows/WindowsTests.cpp
        tests/general/GeneralTests.cpp
)

target_link_libraries(tracker-test tracker-lib)
set_target_properties(tracker-test PROPERTIES OUTPUT_NAME tracker-test)