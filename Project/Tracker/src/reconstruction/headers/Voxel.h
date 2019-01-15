#ifndef TRACKER_LIB_VOXEL_H
#define TRACKER_LIB_VOXEL_H
#include <Eigen/StdVector>

struct Voxel final
{
	Voxel(): m_sdf(-std::numeric_limits<float>::infinity()), m_weight(0), m_free_ctr(0){ }

	float m_sdf;
	unsigned char m_weight;
	int m_free_ctr;
	Vector3f m_color;
	Vector3f m_position;
};

#endif
