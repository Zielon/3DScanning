#ifndef TRACKER_LIB_VOLUME_H
#define TRACKER_LIB_VOLUME_H

#include "../../Eigen.h"
#include "Voxel.h"
#include <opencv2/core/neon_utils.hpp>

class Volume
{
public:
	Volume(Vector3d min, Vector3d max, uint size, uint dim = 1);

	~Volume();

	void forAll(std::function<void(Voxel*, int)> func) const;

	Voxel* getVoxel(int i, int j, int k) const;

	Vector3d position(int i, int j, int k);

private:

	void compute_ddx_dddx();

	int m_size = 50;

	//! Lower left and Upper right corner.
	Vector3d m_min, m_max;

	//! max-min
	Vector3d m_diag;

	double m_ddx, m_ddy, m_ddz;

	double m_dddx, m_dddy, m_dddz;

	std::vector<Voxel*> m_voxels;

	int m_length = 0;

	double m_maxValue, m_minValue;

	uint m_dim;
};

#endif
