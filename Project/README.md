# 3D Scanning and Motion Capture: Final project [Markerless Augmented Reality]

## Installation

## Solution Overview

The solution consist of two main components: a tracker and a visualizer.

### Tracker

The first version of the tracker will estimate the pose of the camera. The final version will also build the mesh representation.

Techniques:
* Iterative Closest Point (ICP): Camera pose estimation.
* Marching cubes: Polygonal mesh extraction.

#### C++ version

Libraries:
* Eigen: linear algebra.
* Ceres: Non-linear optimization.
* FreeImage: image manipulation.
* FLANN: fast nearest neighbor.

#### C# version

### Visualizer

The visualizer component will be implemented in Unity3D.

The first version will be a basic visualizer where a virtual object is placed on top of a planar surface picked by the user, i.e. there is no planar surface detection. The second version will detect the suitable planar surface. The final version will apply complex animations on top of the surface using the mesh representation.

Unity modules:
* UI.
* Graphics.
* Animation.
* Physics.
* Audio.

## References
