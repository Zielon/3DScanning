#ifndef TRACKER_LIB_VOXEL_H
#define TRACKER_LIB_VOXEL_H

enum State
{
	UNSEEN,
	EMPTY,
	SDF
};

struct Voxel final
{
	Voxel(): m_sdf(-std::numeric_limits<unsigned char>::infinity()), m_weight(0), m_state(UNSEEN){ }

	float m_sdf;
	float m_weight;
	State m_state;
};

#endif
