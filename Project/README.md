# 3D Scanning and Motion Capture: Final project [Markerless Augmented Reality]

## Installation

### Visual Studio
In the Visual Studio solutions we are using a few environment paths:

| NAME            | VALUE  														                            |
|-----------------|---------------------------------------------------------------|
|OPENCV_DIR       |`C:\OpenCV\opencv\build\x64\vc15`                              |
|OPENCV_INCLUDE   |`C:\OpenCV\opencv\build\include`                               |
|OPENNI2_INCLUDE64|`C:\Program Files\OpenNI2\Include\`                            |
|OPENNI2_LIB64    |`C:\Program Files\OpenNI2\Lib\`                                |
|PCL_ROOT         |`C:\Program Files\PCL 1.9.1`                                   |
|PCL_INCLUDE      |`C:\Program Files\PCL 1.9.1\include\pcl-1.9`                   |
|SOPHUS_DIR       |`C:\Projects\Sophus-master`                                    |                                     
|EIGEN_DIR        |commit (`83f9cb78d3f455e56653412b7fdb1c0bc3d40ba2`) eigen-git-mirror |

Please set them on your Windows machine!

OpenCV version is 4.0.0. It is important because we are linking: opencvworld400d.lib

The opencvworld400d.dll is already added to \Assets\Plugins in Unity

CUDA 10 Toolkit

The dataset in Unity has to be also in a certain location.
An example path where you have to keep your Freiburg dataset.

`C:\Projects\3DScanning\Project\MarkerlessAR_Unity\Datasets\freiburg\`

### Hardware
* RGB-D Sensor: ASUS Xion Pro (PrimeSense).

### Software
* Operating Systems: Windows 10.
* Programming Languages: C++, C#.
* IDE: Visual Studio 2017.
* Game Engine: Unity Unity 2018.2.19.

### C++ Dependencies
* OpenNI: PrimeSense depth compatible sensors library.
* Eigen: linear algebra library.
* Ceres: non-linear optimization library.
* FreeImage: image manipulation library.
* OpenCV 4.0.0: computer vision library. 
* FLANN: fast nearest neighbor library.

### Unity Modules
* UI.
* Graphics.
* Animation.
* Physics.
* Audio.

### ASUS Xion Pro

The first step is installing the driver associated with the sensor from the official website [0]. The driver can be found by: Support -> Driver & Tools -> Utilities -> *XtionCenter Package*. There is no support for MacOS systems. The next step is installing the OpenNI 2 SDK from the official website [1]. OpenNI 2 includes some sample codes to test the sensor. The *NiViewer* sample was the guideline to integrate the sensor to our project.     
All the files from OpenNI 2's redist directory must be copied to the to working directory (where the project file are located: .vcproj, .vcxproj).   

To change the resolution of the sensor is neccessary to change the values on the file *PS1080.ini*. The following values must be changed:  

* Resolution=1 ([Depth])
* Resolution=1 ([[Image]])
* Resolution=1 ([[IR]])
* UsbInterface=2 ([Device])

Resolution = 1 (VGA -> 640x480), while Resolution = 0 (QVGA -> 320x240).

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
* Multiplatform support (Linux and MacOS).

## References

[0] Xtion PRO: https://www.asus.com/3D-Sensor/Xtion_PRO/.     
[1] OpenNI 2: https://structure.io/openni.  
[2] RGB-D SLAM Dataset and Benchmark: https://vision.in.tum.de/data/datasets/rgbd-dataset.  
[3] Lecture 5: Rigid Surface Tracking & Reconstruction (3D Scanning and Motion Capture/Justus Thies and Angela Dai Slides).  
[4] RGBD streaming by Structure Sensor: https://github.com/occipital/uplink.  
[5] Structure by Occipital: https://structure.io/developers .  
