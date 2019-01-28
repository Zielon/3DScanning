#ifndef TRACKER_LIB_VOXEL_H
#define TRACKER_LIB_VOXEL_H

enum State
{
	UNSEEN = 0,
	EMPTY = 1,
	SDF = 2
};

struct Voxel final
{
	Voxel(): m_sdf(0), m_weight(0), m_state(UNSEEN){ }

	float m_sdf;
	float m_weight;
	int m_state;
};


#endif
