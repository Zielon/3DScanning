#include <utility>
#include "../headers/Volume.h"

Volume::Volume(Vector3d min, Vector3d max, uint size, uint dim) : m_min(std::move(min)), m_max(std::move(max)){
	m_diag = m_max - m_min;
	m_dim = dim;
	m_length = std::pow(m_size, 3);
	m_voxels = std::vector<Voxel*>(m_length, nullptr);
	forAll([this](Voxel* voxel, int index)
	{
		m_voxels[index] = new Voxel();
	});
	compute_ddx_dddx();
}

Volume::~Volume(){
	forAll([this](Voxel* voxel, int index)
	{
		SAFE_DELETE(voxel);
	});
	m_voxels.clear();
}

void Volume::forAll(std::function<void(Voxel*, int)> func) const{
	for (auto x = 0; x < m_length; x++)
		func(m_voxels[x], x);
}

Voxel* Volume::getVoxel(int idx) const{
	if (idx > 0 && idx < m_length)
		return m_voxels[idx];
	return nullptr;
}

void Volume::compute_ddx_dddx(){
	m_ddx = 1.0f / (m_size - 1);
	m_ddy = 1.0f / (m_size - 1);
	m_ddz = 1.0f / (m_size - 1);

	m_dddx = (m_max[0] - m_min[0]) / (m_size - 1);
	m_dddy = (m_max[1] - m_min[1]) / (m_size - 1);
	m_dddz = (m_max[2] - m_min[2]) / (m_size - 1);

	m_diag = m_max - m_min;
}

Voxel* Volume::getVoxel(int x, int y, int z) const{
	const int index = x * m_size * m_size + y * m_size + z;
	if (index < m_length)
		return m_voxels[index];
	return nullptr;
}

/// Returns 3D world position for a given voxel
Vector3f Volume::getWorldPosition(int i, int j, int k){
	Vector3f coordinates;

	coordinates[0] = m_min[0] + (m_max[0] - m_min[0]) * (double(i) * m_ddx);
	coordinates[1] = m_min[1] + (m_max[1] - m_min[1]) * (double(j) * m_ddy);
	coordinates[2] = m_min[2] + (m_max[2] - m_min[2]) * (double(k) * m_ddz);

	return coordinates;
}
