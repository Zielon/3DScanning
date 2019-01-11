#include "../headers/Fusion.h"
#include "../headers/Mesh.h"
#include "../headers/MarchingCubes.h"

#include <direct.h>
#include <io.h>
#include <math.h>

Fusion::Fusion(CameraParameters camera_parameters) : m_camera_parameters(camera_parameters){
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

inline Vector3i round(const Vector3f& point){
	return Vector3i(round(point.x()), round(point.y()), round(point.z()));
}

inline float getTruncation(float depth){
	if (depth > 1) return 1;
	if (depth < -1) return -1;
	return depth;
}

void Fusion::integrate(PointCloud* cloud){

	const auto cameraToWorld = cloud->m_pose_estimation;
	const auto worldToCamera = cameraToWorld.inverse();

	const auto rotation = worldToCamera.block(0, 0, 3, 3);
	const auto translation = worldToCamera.block(0, 3, 3, 1);
	const auto frustum_box = computeFrustumBounds(cameraToWorld);

	#pragma omp parallel
	for (unsigned int z = frustum_box.m_min_z; z < frustum_box.m_max_z; z++)
		for (unsigned int y = frustum_box.m_min_y; y < frustum_box.m_max_y; y++)
			for (unsigned int x = frustum_box.m_min_x; x < frustum_box.m_max_x; x++)
			{
				Voxel* voxel = m_volume->getVoxel(x, y, z);

				// Transform from the cell world to the camera world
				Vector3f p = rotation * m_volume->getWorldPosition(x, y, z) + translation;

				// Project into a depth image
				p = skeletonToDepth(p);

				// Pixels space
				auto pi = round(p);

				float d = cloud->depthImage(pi.x(), pi.y());

				// Depth was not found
				if (d == INFINITY) continue;

				// Update free space counter if voxel is in front of observation
				if (p.z() < d)
					voxel->m_free_ctr++;

				// Positive in front of the observation
				const float sdf = d - p.z();
				const float truncation = getTruncation(d);
				const float weight = voxel->m_weight;

				if (sdf > -truncation)
				{
					voxel->m_sdf = (voxel->m_sdf * weight + sdf * m_weight_update) / (weight + m_weight_update);
					voxel->m_weight = std::min(int(weight) + int(m_weight_update),
					                           int(std::numeric_limits<unsigned char>::max()));
					voxel->m_position = Vector3f(x, y, z);
				}

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

Vector3f Fusion::skeletonToDepth(Vector3f point) const{
	float x = m_camera_parameters.m_focal_length_X * point.x() / point.z() + m_camera_parameters.m_cX;
	float y = m_camera_parameters.m_focal_length_Y * point.y() / point.z() + m_camera_parameters.m_cY;
	return Vector3f(x, y, 0);
}

FrustumBox Fusion::computeFrustumBounds(Matrix4f cameraToWorld) const{

	const auto rotation = cameraToWorld.block(0, 0, 3, 3);
	const auto translation = cameraToWorld.block(0, 3, 3, 1);

	// Assuming that a camera is placed in (0,0,0)
	Vector3f cameraWorld = rotation * Vector3f(0, 0, 0) + translation;

	FrustumBox box;

	box.m_min_x = 0;
	box.m_min_y = 0;
	box.m_min_z = 0;

	box.m_max_x = m_volume->m_size;
	box.m_max_y = m_volume->m_size;
	box.m_max_z = m_volume->m_size;

	return box;
}
