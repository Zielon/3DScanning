cmake_minimum_required(VERSION 3.10)
project(tracker VERSION 1.0.0 DESCRIPTION "A library for tracking")

# Compiler settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")

# Build directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# ---------> REQUIRED PACKAGES

if (UNIX)
    find_package(Eigen3 3.3 REQUIRED)
    include_directories(${EIGEN3_INCLUDE_DIR})

    find_package(OpenCV REQUIRED)
    include_directories(${OPENCV_INCLUDE_DIR})

    find_package(Ceres REQUIRED)
    include_directories(${CERES_INCLUDE_DIRS})
    include_directories(include ${PROJECT_BINARY_DIR})
endif (UNIX)

# -----------------------------

# ------------Compile dll library (deploy)
add_library(
        tracker SHARED
        Tracker.cpp
        ICP.cpp
        ExportDLL.h
)

#--------------Compile library (Linux and Apple development)
if (UNIX)
    target_link_libraries(
            tracker
            freeimage
            opencv_core
            opencv_highgui
            opencv_features2d
            opencv_imgcodecs
            ${CERES_LIBRARIES}
    )
endif (UNIX)

target_link_libraries(
        tracker
)

add_executable(
        tracker-bin
        main.cpp
)

target_link_libraries(tracker-bin tracker)
set_target_properties(tracker-bin PROPERTIES OUTPUT_NAME tracker)