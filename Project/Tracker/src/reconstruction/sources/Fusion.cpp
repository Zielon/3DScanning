#include "../headers/Fusion.h"

Fusion::Fusion(CameraParameters camera_parameters) : m_camera_parameters(camera_parameters){
	m_volume = new Volume(Vector3d(-0.1, -0.1, -0.1), Vector3d(1.1, 1.1, 1.1), m_volume_size, 1);
}

Fusion::~Fusion(){
	m_consumer->stop();
	m_consumer_thread.join();

	SAFE_DELETE(m_volume);
	SAFE_DELETE(m_buffer);
	SAFE_DELETE(m_consumer);
}

/// Consumes point clouds from a buffer and 
/// produces a mesh using SFD implicit functions
void Fusion::consume(){
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
void Fusion::produce(PointCloud* cloud) const{
	m_buffer->add(cloud);
}

void Fusion::integrate(PointCloud* cloud) const{
	#pragma omp parallel
	for (unsigned int x = 0; x < m_volume_size; x++)
		for (unsigned int y = 0; y < m_volume_size; y++)
			for (unsigned int z = 0; z < m_volume_size; z++)
			{
				Vector3f surface = m_volume->getWorldPosition(x, y, z);
				Voxel* voxel = m_volume->getVoxel(x, y, z);
				int index = cloud->getClosestPoint(surface);

				if (!voxel || index < 0) continue;

				Vector3f point = cloud->getPoints()[index];
				Vector3f normal = cloud->getNormals()[index];

				if (index == cloud->getPoints().size())
					voxel->m_sdf = 0.0f;
				
				voxel->m_sdf = (surface - point).transpose() * normal;
				voxel->m_free_ctr++;
			}

	//SAFE_DELETE(cloud);
}
