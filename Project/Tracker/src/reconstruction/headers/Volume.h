#ifndef TRACKER_LIB_VOLUME_H
#define TRACKER_LIB_VOLUME_H

#include "../../Eigen.h"
#include "Voxel.h"
#include <opencv2/core/neon_utils.hpp>

struct Size
{
	Size(float height, float width, float depth) : m_width(width), m_height(height), m_depth(depth){ }

	float m_width;
	float m_height;
	float m_depth;
};

class Volume
{
public:
	Volume(Size min, Size max, uint size, uint dim = 1);

	~Volume();

	void forAll(std::function<void(Voxel*, int)> func) const;

	Voxel* getVoxel(int i, int j, int k) const;

	Voxel* getVoxel(Vector3i position) const;

	Voxel* getVoxel(int idx) const;

	Vector3f getWorldPosition(Vector3i position);

	Vector3i getGridPosition(Vector3f position);

	float m_voxel_size;

	int m_size;

private:

	//! Lower left and Upper right corner.
	Vector3d m_min, m_max;

	//! max-min
	Vector3d m_diag;

	double m_ddx, m_ddy, m_ddz;

	Voxel* m_voxels;

	int m_length = 0;

	double m_maxValue, m_minValue;

	uint m_dim;
};

#endif
