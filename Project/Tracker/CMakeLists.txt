cmake_minimum_required(VERSION 3.10)
project(tracker VERSION 1.0.0 DESCRIPTION "A library for tracking")

# Deployment scenario
set (DEPLOYMENT TRUE)

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

set(LIBS "${CMAKE_SOURCE_DIR}/dependencies")
set(OPEN_NI_DIR "C:/Users/Lukas/Documents/OpenNI")


if (WIN32)

	if (DEPLOYMENT)
		include_directories(${LIBS}/opencv/include)
		#include_directories(${LIBS}/OpenNI/Include)
		include_directories(${OPEN_NI_DIR}/Include)

	else ()
		include_directories(${LIBS}/opencv/include)
	endif()

endif (WIN32)

# ------------Compile dll library (deploy)
if (DEPLOYMENT)

	add_library(
        tracker SHARED
        Tracker.cpp
		XtionStreamReader.cpp
		ICP.cpp
        ExportDLL.cpp
	)

else()

	add_library(
        tracker SHARED
        Tracker.cpp
        DatasetVideoStreamReader.cpp
		ICP.cpp
        ExportDLL.cpp
	)

endif()

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

#--------------Compile library (Windows development)
if (WIN32)

	if (DEPLOYMENT)

		target_link_libraries(tracker PUBLIC
			${LIBS}/opencv/libs/opencv.lib
			${LIBS}/OpenNI/Lib/openNI.lib
			#${OPEN_NI_DIR}/Lib/openNI.lib
		)

	else()

		target_link_libraries(
			tracker
			${LIBS}/opencv/libs/opencv.lib
		)

	endif()

endif (WIN32)

if (DEPLOYMENT)

	add_executable(
        tracker-bin
        main.cpp
	)

else()

	add_executable(
        tracker-bin
        main.cpp
        Tests/WindowsTests.cpp
	)

endif()

target_link_libraries(tracker-bin tracker)
set_target_properties(tracker-bin PROPERTIES OUTPUT_NAME tracker)