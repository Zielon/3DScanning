#include "../headers/Fusion.h"
#include "../headers/Mesh.h"
#include "../headers/MarchingCubes.h"

#include <direct.h>
#include <cmath>
#include "../../concurency/headers/ThreadManager.h"

Fusion::Fusion(CameraParameters camera_parameters) : m_camera_parameters(camera_parameters){
	initialize();
}

Fusion::~Fusion(){
	std::for_each(m_consumers.begin(), m_consumers.end(),
	              [](auto consumer){
		              consumer->stop();
	              });

	ThreadManager::waitForAll(m_consumer_threads);

	std::for_each(m_consumers.begin(), m_consumers.end(), [](auto consumer){
		SAFE_DELETE(consumer);
	});
	SAFE_DELETE(m_volume);
	SAFE_DELETE(m_buffer);
}

inline Vector3i round(const Vector3f& point){
	return Vector3i(std::round(point.x()), std::round(point.y()), std::round(point.z()));
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

	if (!frustum_box.m_is_valid) return;

	for (unsigned int z = frustum_box.m_min_z; z < frustum_box.m_max_z; z++)
		for (unsigned int y = frustum_box.m_min_y; y < frustum_box.m_max_y; y++)
			for (unsigned int x = frustum_box.m_min_x; x < frustum_box.m_max_x; x++)
			{
				// Transform from the cell world to the camera world
				Vector3f cell = rotation * m_volume->getWorldPosition(Vector3i(x, y, z)) + translation;

				// Project into a depth image
				cell = reproject(cell);

				// Pixels space
				auto pixels = round(cell);

				float depth = cloud->getDepthImage(pixels.x(), pixels.y());

				// Depth was not found
				if (depth == INFINITY) continue;

				m_mutex.lock();

				Voxel* voxel = m_volume->getVoxel(x, y, z);

				// Update free space counter if voxel is in the front of observation
				if (cell.z() < depth)
					voxel->m_free_ctr++;

				// Positive in front of the observation
				const float sdf = depth - cell.z();
				const float truncation = getTruncation(depth);
				const float weight = voxel->m_weight;

				if (sdf > -truncation)
				{
					voxel->m_sdf = (voxel->m_sdf * weight + sdf * m_weight_update) / (weight + m_weight_update);
					voxel->m_weight = std::min(int(weight) + int(m_weight_update),
					                           int(std::numeric_limits<unsigned char>::max()));
					voxel->m_position = Vector3f(x, y, z);
				}

				m_weight_update += weight;

				m_mutex.unlock();
			}

	SAFE_DELETE(cloud);
}

void Fusion::save(string name){
	std::for_each(m_consumers.begin(), m_consumers.end(),
	              [](auto consumer){
		              consumer->stop();
	              });

	ThreadManager::waitForAll(m_consumer_threads);

	Mesh mesh;

	for (unsigned int x = 0; x < m_volume->m_size - 1; x++)
		for (unsigned int y = 0; y < m_volume->m_size - 1; y++)
			for (unsigned int z = 0; z < m_volume->m_size - 1; z++)
				ProcessVolumeCell(m_volume, x, y, z, 0.9f, &mesh);

	mesh.save(name);
}

bool Fusion::isFinished() const{
	return m_buffer->isEmpty();
}

/// Buffer has a certain capacity when it is exceeded 
/// this method will block the execution
void Fusion::produce(PointCloud* cloud) const{
	m_buffer->add(cloud);
}

/// Consumes point clouds from a buffer and 
/// produces a mesh using SFD implicit functions
void Fusion::consume(){

	for (int i = 0; i < NUMBER_OF_CONSUMERS; i++)
	{
		auto consumer = new Consumer<PointCloud*>(m_buffer);
		m_consumers.emplace_back(consumer);
		m_consumer_threads.emplace_back([this, consumer](){
			consumer->run([this](PointCloud* cloud){
				this->integrate(cloud);
			});
		});
	}
}

void Fusion::initialize(){
	m_volume = new Volume(Vector3d(-5, -5, -5), Vector3d(5, 5, 5), 200, 1);
	m_buffer = new Buffer<PointCloud*>();
}

Vector3f Fusion::reproject(Vector3f point) const{
	float x = m_camera_parameters.m_focal_length_X * point.x() / point.z() + m_camera_parameters.m_cX;
	float y = m_camera_parameters.m_focal_length_Y * point.y() / point.z() + m_camera_parameters.m_cY;
	return Vector3f(x, y, 0);
}

FrustumBox Fusion::computeFrustumBounds(Matrix4f cameraToWorld) const{

	const auto translation = cameraToWorld.block(0, 3, 3, 1);

	// Assuming that a camera is placed in (0,0,0)
	Vector3f cameraWorld = Vector3f(0, 0, 0) + translation;
	Vector3i cameraGrid = m_volume->getGridPosition(cameraWorld);
	Voxel* cameraVoxel = m_volume->getVoxel(cameraGrid);

	FrustumBox box;

	// Skip if camera is not in our mapping space
	if (!cameraVoxel)
	{
		box.m_is_valid = false;
		return box;
	}

	// Camera has a valid position
	box.m_is_valid = true;

	// TODO Calculate bounds of the viewing frustum

	box.m_min_x = 0;
	box.m_min_y = 0;
	box.m_min_z = 0;

	box.m_max_x = m_volume->m_size;
	box.m_max_y = m_volume->m_size;
	box.m_max_z = m_volume->m_size;

	return box;
}
