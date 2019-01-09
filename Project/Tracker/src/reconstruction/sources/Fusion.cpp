#include "../headers/Fusion.h"

Fusion::Fusion(CameraParameters camera_parameters) : m_camera_parameters(camera_parameters){
	m_voxles_space = vector<vector<vector<Voxel*>>>(
		m_size, vector<vector<Voxel*>>(m_size, vector<Voxel*>(m_size, new Voxel())));

	forAll([](Voxel* cell, Vector3f position)
	{
		cell->m_position = position;
	});
}

Fusion::~Fusion(){
	forAll([](Voxel* cell, Vector3f _)
	{
		SAFE_DELETE(cell);
	});
	m_voxles_space.clear();
}

void Fusion::integrate(const PointCloud& cloud, Matrix4f& pose){

	auto worldToCamera = pose.inverse();

}

vector<vector<vector<Voxel*>>>& Fusion::getTSDF(){
	return m_voxles_space;
}

/// Parallel execution therefore the callback function
/// has to be thread-safe
void Fusion::forAll(const function<void(Voxel*, Vector3f)> func){
	#pragma omp parallel
	for (auto x = 0; x < m_size; x++)
		for (auto y = 0; y < m_size; y++)
			for (auto z = 0; z < m_size; z++)
				func(m_voxles_space[x][y][z], Vector3f(x, y, z));
}

Voxel* Fusion::get(int i, int j, int k){
	if (i < m_size && j < m_size && k < m_size)
		return m_voxles_space[i][j][k];

	throw new exception("Out of range!");
}
