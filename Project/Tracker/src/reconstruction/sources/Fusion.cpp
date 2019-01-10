#include "../headers/Fusion.h"

Fusion::Fusion(CameraParameters camera_parameters) : m_camera_parameters(camera_parameters){
	m_volume = new Volume(Vector3d(-0.1, -0.1, -0.1), Vector3d(1.1, 1.1, 1.1), m_volume_size, 1);
}

Fusion::~Fusion(){
	delete m_volume;
}

void Fusion::startConsuming(){
	m_consumer_thread = std::thread([this]()
	{
		// It will block the thread in the case of an empty buffer
		m_consumer->run([this](PointCloud* cloud)
		{
			this->integrate(cloud);
		});
	});
}

/// Buffer has a certain capacity when it is exceeded 
/// this method will block the execution
void Fusion::addToBuffer(PointCloud* cloud) const{
	m_buffer->add(cloud);
}

void Fusion::integrate(PointCloud* cloud){
	#pragma omp parallel
	for (unsigned int x = 0; x < m_volume_size; x++)
		for (unsigned int y = 0; y < m_volume_size; y++)
			for (unsigned int z = 0; z < m_volume_size; z++)
			{
				Voxel* voxel = m_volume->getVoxel(x, y, z);

				if (!voxel) continue;

				voxel->m_distance = 0;
			}
}
