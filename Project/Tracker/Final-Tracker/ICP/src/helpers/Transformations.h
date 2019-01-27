#ifndef TRACKER_LIB_TRANSFORMATIONS_H
#define TRACKER_LIB_TRANSFORMATIONS_H

#include "../Eigen.h"
#include "../reconstruction/headers/SystemParameters.h"
#include "../marshaling/__Mesh.h"

//ICP Parameters
#define max_distance 0.0003f

class Transformations
{
public:

	static Vector3f backproject(float x, float y, float depth, SystemParameters camera_parameters){
		Vector3f point;
		point[0] = (x - camera_parameters.m_cX) / camera_parameters.m_focal_length_X * depth;
		point[1] = (y - camera_parameters.m_cY) / camera_parameters.m_focal_length_Y * depth;
		point[2] = depth;
		return point;
	}

	static Vector3f reproject(Vector3f point, SystemParameters camera_parameters){
		float x = camera_parameters.m_focal_length_X * point.x() / point.z() + camera_parameters.m_cX;
		float y = camera_parameters.m_focal_length_Y * point.y() / point.z() + camera_parameters.m_cY;
		return Vector3f(x, y, point.z());
	}
};

#endif
