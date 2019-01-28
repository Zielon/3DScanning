#include <utility>
#include "../headers/Volume.h"

/// Size vector elements:
/// y -> width
/// x -> height
///
Volume::Volume(Size min, Size max, uint size, uint dim, bool allocMemory){
	m_dim = dim;
	m_size = size;
	m_length = std::pow(m_size, 3);
	m_voxel_size = (max.m_width - min.m_width) / float(size);

	// depth, width, height
	m_min = Vector3d(min.m_depth, min.m_width, min.m_height);
	m_max = Vector3d(max.m_depth, max.m_width, max.m_height);

	if (allocMemory)
	{
		m_voxels = new Voxel[m_length];
	}

}

Volume::~Volume(){
	delete[] m_voxels;
}

void Volume::forAll(std::function<void(Voxel*, int)> func) const{
	#pragma omp parallel for
	for (auto x = 0; x < m_length; x++)
		func(m_voxels + x, x);
}

Voxel* Volume::getVoxel(int idx) const{
	if (idx >= 0 && idx < m_length)
		return m_voxels + idx;
	return nullptr;
}

Voxel* Volume::getVoxel(int x, int y, int z) const{
	const int index = x * m_size * m_size + y * m_size + z;
	return m_voxels + index;
}

Voxel* Volume::getVoxel(Vector3i position) const{
	return getVoxel(position[0], position[1], position[2]);
}

/// Returns 3D world position for a given voxel
Vector3f Volume::getWorldPosition(Vector3i position){

	const auto invScaling = (m_size - 1.0);
	Vector3f world = (m_min + (m_max - m_min).cwiseProduct(position.cast<double>() / invScaling)).cast<float>();

	return world;
}

Vector3i Volume::getGridPosition(Vector3f position){

	const auto invScaling = (m_size - 1.0);
	Vector3i grid = (invScaling * (position.cast<double>() - m_min).cwiseQuotient(m_max - m_min))
	                .array().round().matrix().cast<int>();

	return grid;
}
