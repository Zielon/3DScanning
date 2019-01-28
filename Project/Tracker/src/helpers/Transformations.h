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

	static Vector3f backproject(float x, float y, float depth, SystemParameters system_parameters){
		Vector3f point;
		point[0] = (x - system_parameters.m_cX) / system_parameters.m_focal_length_X * depth;
		point[1] = (y - system_parameters.m_cY) / system_parameters.m_focal_length_Y * depth;
		point[2] = depth;
		return point;
	}

	static Vector3f reproject(Vector3f point, SystemParameters system_parameters){
		float x = system_parameters.m_focal_length_X * point.x() / point.z() + system_parameters.m_cX;
		float y = system_parameters.m_focal_length_Y * point.y() / point.z() + system_parameters.m_cY;
		return Vector3f(x, y, point.z());
	}
};

#endif
