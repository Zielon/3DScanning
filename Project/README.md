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

### ASUS Xion Pro

The first step is installing the driver associated with the sensor from the official website [0]. The driver can be found by: Support -> Driver & Tools -> Utilities -> *XtionCenter Package*. There is no support for MacOS systems. The next step is installing the OpenNI 2 SDK from the official website [1]. OpenNI 2 includes some sample codes to test the sensor. The *NiViewer* sample was the guideline to integrate the sensor to our project. Finally, a small change is required in order to use the OpenNI 2 SDK with Visual Studio 2017. The line 58 of the *XnPlatform.h* file must be commented out.

Error line:  
> #error Xiron Platform Abstraction Layer - Win32 - Microsoft Visual Studio versions above 2010 (10.0) are not supported!

## Solution Overview

The solution consists of four main components: a RGB-D stream, a camera tracker, a reconstruction module and an AR animation module.

### RGB-D Stream

This component is responsible to obtain the color map and depth map of each frame, which will be used for the other components. There are two types of RGB-D Stream sources: datasets and sensors.

The *DatasetVideoStreamReader* class implements all the functionalities to import color maps and depth maps of recorded camera data from the *TUM RGB-D SLAM Dataset* [2].

The *XtionStreamReader* class implements all the functionalities to import color maps and depth maps in real time from the ASUS Xion Pro sensor.  

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

## Future Development

Future tasks:
* iPad with Structure Sensor Deployment using uplink library [4] or Structure SDK for iOS [5].

## References

[0] Xtion PRO: https://www.asus.com/3D-Sensor/Xtion_PRO/.   
[1] OpenNI 2: https://structure.io/openni.
[2] RGB-D SLAM Dataset and Benchmark: https://vision.in.tum.de/data/datasets/rgbd-dataset.
[3] Lecture 5: Rigid Surface Tracking & Reconstruction (3D Scanning and Motion Capture/Justus Thies and Angela Dai Slides).
[4] RGBD streaming by Structure Sensor: https://github.com/occipital/uplink.
[5] Structure by Occipital: https://structure.io/developers .