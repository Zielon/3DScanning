#include <utility>
#include "../headers/Volume.h"

Volume::Volume(Vector3d min, Vector3d max, uint size, uint dim) : m_min(std::move(min)), m_max(std::move(max)){
	m_dim = dim;
	m_size = size;
	m_length = std::pow(m_size, 3);
	m_voxels = std::vector<Voxel*>(m_length, nullptr);
	forAll([this](Voxel*, int index){
		m_voxels[index] = new Voxel();
	});
}

Volume::~Volume(){
	forAll([this](Voxel* voxel, int index){
		SAFE_DELETE(voxel);
	});
	m_voxels.clear();
}

void Volume::forAll(std::function<void(Voxel*, int)> func) const{
	for (auto x = 0; x < m_length; x++)
		func(m_voxels[x], x);
}

Voxel* Volume::getVoxel(int idx) const{
	if (idx >= 0 && idx < m_length)
		return m_voxels[idx];
	return nullptr;
}

Voxel* Volume::getVoxel(int x, int y, int z) const{
	const int index = x * m_size * m_size + y * m_size + z;
	if (index < m_length)
		return m_voxels[index];
	return nullptr;
}

Voxel* Volume::getVoxel(Vector3i position) const{
	return getVoxel(position[0], position[1], position[2]);
}

/// Returns 3D world position for a given voxel
Vector3f Volume::getWorldPosition(Vector3i position) {
	Vector3f world;

	const auto invScaling = (m_size - 1.0);
	world = (m_min + (m_max - m_min).cwiseProduct(position.cast<double>() / invScaling)).cast<float>();

	return world;
}

Vector3i Volume::getGridPosition(Vector3f position) {
	Vector3i grid;

	const auto invScaling = (m_size - 1.0);
	grid = (invScaling * (position.cast<double>() - m_min).cwiseQuotient(m_max - m_min)).array().round().matrix().cast<int>();

	return grid;
}
