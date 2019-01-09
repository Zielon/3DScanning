#ifndef TRACKER_LIB_VOXEL_H
#define TRACKER_LIB_VOXEL_H
#include <Eigen/StdVector>

struct Voxel final
{
	float m_distance;
	float m_weight;
	Eigen::Vector3f m_color;
};

#endif
