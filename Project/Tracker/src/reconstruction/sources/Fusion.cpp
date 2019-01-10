#include "../headers/Fusion.h"
#include "../headers/Mesh.h"
#include "../headers/MarchingCubes.h"

#include <direct.h>
#include <io.h>

Fusion::Fusion(CameraParameters camera_parameters) : m_camera_parameters(camera_parameters){
	initialize();
}

Fusion::Fusion(int width, int height, int pixelSteps) : m_height(height), m_width(width), m_pixelSteps(pixelSteps){
	initialize();
}

Fusion::~Fusion(){
	m_consumer->stop();
	if (m_consumer_thread.joinable())
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

void Fusion::integrate(PointCloud* cloud) {

	Matrix4f pose = cloud->m_pose_estimation;

	const auto rotation = pose.block(0, 0, 3, 3);
	const auto translation = pose.block(0, 3, 3, 1);

	#pragma omp parallel
	for (unsigned int x = 0; x < m_volume->m_size; x++)
		for (unsigned int y = 0; y < m_volume->m_size; y++)
			for (unsigned int z = 0; z < m_volume->m_size; z++)
			{
				Vector3f cell = m_volume->getWorldPosition(x, y, z);
				Voxel* voxel = m_volume->getVoxel(x, y, z);
				int index = cloud->getClosestPoint(cell);

				if (!voxel || index < 0 || index == cloud->getPoints().size()) continue;

				Vector3f point = cloud->getPoints()[index];

				point = rotation * point + translation;

				// Update free space counter if voxel is in front of observation
				if (cell.z() < point.z())
					voxel->m_free_ctr++;

				// Positive in front of the observation
				const float sdf = point.z() - cell.z();
				const float weight = voxel->m_weight;

				voxel->m_sdf = (voxel->m_sdf * weight + sdf * m_weight_update) / (weight + m_weight_update);
				voxel->m_weight = std::min(int(weight) + int(m_weight_update), int(std::numeric_limits<unsigned char>::max()));
				voxel->m_position = Vector3f(x, y, z);

				m_weight_update += weight;
			}

	SAFE_DELETE(cloud);
}

void Fusion::save(string name){
	m_consumer->stop();
	if (m_consumer_thread.joinable())
		m_consumer_thread.join();

	string folder = "test_meshes";

	_mkdir(folder.data());

	Mesh mesh;

	for (unsigned int x = 0; x < m_volume->m_size - 1; x++)
		for (unsigned int y = 0; y < m_volume->m_size - 1; y++)
			for (unsigned int z = 0; z < m_volume->m_size - 1; z++)
				ProcessVolumeCell(m_volume, x, y, z, 0.00f, &mesh);

	mesh.WriteMesh(folder + "\\" + name);
}

void Fusion::initialize(){
	m_volume = new Volume(Vector3d(-0.5, -0.5, -0.5), Vector3d(1.5, 1.5, 1.5), 150, 1);
	m_buffer = new Buffer<PointCloud*>();
	m_consumer = new Consumer<PointCloud*>(m_buffer);
}
