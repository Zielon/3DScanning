#include <utility>
#include "../headers/Volume.h"

Volume::Volume(Vector3d min, Vector3d max, uint size, uint dim) : m_min(std::move(min)), m_max(std::move(max)),
                                                                  m_voxels(nullptr){
	m_diag = m_max - m_min;
	m_dim = dim;
	m_length = std::pow(m_size, 3);
	m_voxels = new std::vector<Voxel*>(m_length, new Voxel());
	compute_ddx_dddx();
}

Volume::~Volume(){
	forAll([](Voxel* cell)
	{
		SAFE_DELETE(cell);
	});

	m_voxels->clear();
	delete m_voxels;
}

/// Parallel execution therefore the callback function
/// has to be thread-safe
void Volume::forAll(const std::function<void(Voxel*)> func) const{
	#pragma omp parallel
	for (auto x = 0; x < m_length; x++)
		func(m_voxels->operator[](x));
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
	if(index < m_length)
		return m_voxels->operator[](index);
	return nullptr;
}

/// Returns 3D world position for a given voxel
Vector3d Volume::position(int i, int j, int k){
	Vector3d coordinates{};

	coordinates[0] = m_min[0] + (m_max[0] - m_min[0]) * (double(i) * m_ddx);
	coordinates[1] = m_min[1] + (m_max[1] - m_min[1]) * (double(j) * m_ddy);
	coordinates[2] = m_min[2] + (m_max[2] - m_min[2]) * (double(k) * m_ddz);

	return coordinates;
}
