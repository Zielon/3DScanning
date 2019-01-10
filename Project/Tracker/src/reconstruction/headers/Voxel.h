#ifndef TRACKER_LIB_VOXEL_H
#define TRACKER_LIB_VOXEL_H
#include <Eigen/StdVector>

struct Voxel final
{
	Voxel(): m_sdf(INFINITY), m_weight(0), m_free_ctr(0), m_color(Vector3f()){ }

	float m_sdf;
	float m_weight;
	int m_free_ctr;
	Vector3f m_color;
	Vector3f m_position;
};

#endif
