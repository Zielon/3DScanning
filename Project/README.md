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

## Solution Overview

The solution consists of four main components: a RGB-D stream, a camera tracker, a reconstruction module and an AR animation module.

### RGB-D Stream

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

