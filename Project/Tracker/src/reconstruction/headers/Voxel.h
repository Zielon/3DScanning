#ifndef TRACKER_LIB_VOXEL_H
#define TRACKER_LIB_VOXEL_H

#define SDF_MIN -10000000000.f

struct Voxel final
{
	Voxel(): m_sdf(SDF_MIN), m_weight(0), m_ctr(0){ }

	float m_sdf;
	float m_weight;
	int m_ctr;
	Vector3f m_color;
};

#endif
