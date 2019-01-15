#ifndef TRACKER_LIB_TRANSFORMATIONS_H
#define TRACKER_LIB_TRANSFORMATIONS_H

#include "../Eigen.h"
#include "../reconstruction/headers/CameraParameters.h"

class Transformations
{
public:

	static Vector3f backproject(float x, float y, float depth, CameraParameters camera_parameters){
		Vector3f point;
		point[0] = (x - camera_parameters.m_cX) / camera_parameters.m_focal_length_X * depth;
		point[0] = (y - camera_parameters.m_cY) / camera_parameters.m_focal_length_Y * depth;
		point[0] = depth;
		return point;
	}

	static Vector3f reproject(Vector3f point, CameraParameters camera_parameters){
		float x = camera_parameters.m_focal_length_X * point.x() / point.z() + camera_parameters.m_cX;
		float y = camera_parameters.m_focal_length_Y * point.y() / point.z() + camera_parameters.m_cY;
		return Vector3f(x, y, point.z());
	}
};

#endif
