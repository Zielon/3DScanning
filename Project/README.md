# 3D Scanning and Motion Capture: Final project [Markerless Augmented Reality]

## Installation

### Hardware
* RGB-D Sensor: ASUS Xion Pro (PrimeSense).

### Software
* Operating Systems: Windows, Linux and MacOS.
* Programming Languages: C++, C#.
* Game Engine: Unity.

### C++ Dependencies
* OpenNI: PrimeSense depth compatible sensors library.
* Eigen: linear algebra library.
* Ceres: non-linear optimization library.
* FreeImage: image manipulation library.
* FLANN: fast nearest neighbor library.

### Unity Modules
* UI.
* Graphics.
* Animation.
* Physics.
* Audio.

### RGB-D Sensor

First, the driver of the ASUS Xion Pro sensor must be installed. The driver is found automatically in Windows. As a result, an application XtionCenter is installed. This application provides examples and games using the * ASUS Xion Pro *. Second, OpenNI 2 must be installed. OpenNI 2 website includes a Windows installer. OpenNI 2 also provides some samples. Finally, the file *XnPlatform.h* must be changed in order to use OpenNI 2 library on Visual Studio 2017. The line 58 should be commented out. 

#error Xiron Platform Abstraction Layer - Win32 - Microsoft Visual Studio versions above 2010 (10.0) are not supported!

## Solution Overview

The solution consists of four main components: a RGB-D stream, a camera tracker, a reconstruction module and an AR animation module.

### RGB-D Stream

The goal of this component is obtaining the RGB and Depth data, which will be used for the other components. There are two RGB-D streams: datasets and sensors.

The recorded camera data can be obtained from the TUM RGB-D SLAM Dataset: https://vision.in.tum.de/data/datasets/rgbd-dataset. These datasets can be processed using the *DatasetVideoStreamReader* class.

The *XtionStreamReader* class implements all the functionalities to obtain the final color and depth frames from the ASUS Xion Pro sensor.

### Camera Tracker

This component will estimate the pose of the camera using the color map and depth map given by the RGB-D Stream. The output of the component is the pose transformation of each frame.

Techniques:
* Iterative Closest Point (ICP): Camera pose estimation.

### Reconstruction

This component will estimate the mesh representation of the scene using the information given by the RGB-D stream and the camera tracker.

Techniques:
* Volumetric fusion: shared model using Signed Distance Field (SDF) representation.
* Marching cubes: Polygonal mesh extraction of the SDF.

### AR Animation

This component will be implemented in Unity3D.

The first version will be a basic AR animation where a virtual object is placed on top of a planar surface picked by the user, i.e. there is no planar surface detection. The second version will detect the suitable planar surface. The final version will apply complex animations on top of the surface using the mesh representation.

## References

