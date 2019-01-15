#include "../headers/Fusion.h"
#include "../headers/Mesh.h"
#include "../headers/MarchingCubes.h"

#include <direct.h>
#include <cmath>
#include <algorithm>
#include <utility>
#include "../../concurency/headers/ThreadManager.h"
#include "../../debugger/headers/ProgressBar.hpp"
#include "../../helpers/Transformations.h"

Fusion::Fusion(CameraParameters camera_parameters) : m_camera_parameters(std::move(camera_parameters)){
	initialize();
}

Fusion::~Fusion(){
	stopConsumers();
	SAFE_DELETE(m_buffer);
	ThreadManager::waitForAll(m_consumer_threads);
	std::for_each(m_consumers.begin(), m_consumers.end(), [](auto consumer){
		SAFE_DELETE(consumer);
	});
	SAFE_DELETE(m_volume);
}

inline Vector3i round(const Vector3f& point){
	return Vector3i(std::round(point.x()), std::round(point.y()), std::round(point.z()));
}

inline float getTruncation(float depth){
	if (depth > 1) return 1;
	if (depth < -1) return -1;
	return depth;
}

void Fusion::save(string name) const{

	wait();

	Mesh mesh;

	for (int x = 0; x < m_volume->m_size - 1; x++)
		for (int y = 0; y < m_volume->m_size - 1; y++)
			for (int z = 0; z < m_volume->m_size - 1; z++)
				ProcessVolumeCell(m_volume, x, y, z, -0.15f, &mesh);

	mesh.save(name);
}

void Fusion::initialize(){
	m_volume = new Volume(Vector3d(-3, -2, -2), Vector3d(3, 2, 2), 200, 1);
	m_buffer = new Buffer<PointCloud*>();
}

void Fusion::wait() const{
	const int size = m_buffer->size();
	if (size == 0) return;
	Verbose::message("Waiting for consumers... [ " + std::to_string(size) + " frames ]");
	ProgressBar bar(size, 60);

	while (!m_buffer->isEmpty())
	{
		bar.set(size - m_buffer->size());
		bar.display();
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
	}

	bar.set(size - m_buffer->size());
	bar.done();
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

int Fusion::clamp(float value) const{
	const auto max = float(m_volume->m_size);
	return int(max(0.f, min(max, value)));
}

Vector3i Fusion::clamp(Vector3i value) const{
	return Vector3i(clamp(value.x()), clamp(value.y()), clamp(value.z()));
}

void Fusion::stopConsumers(){
	for (auto& consumer : m_consumers)
		consumer->stop();
}

void Fusion::integrate(PointCloud* cloud){
	const auto cameraToWorld = cloud->m_pose_estimation;
	const auto worldToCamera = cameraToWorld.inverse();

	const auto rotation = worldToCamera.block(0, 0, 3, 3);
	const auto translation = worldToCamera.block(0, 3, 3, 1);
	const auto frustum_box = computeFrustumBounds(cameraToWorld, cloud->m_camera_parameters);

	for (unsigned int z = frustum_box.m_min.z(); z < frustum_box.m_max.z(); z++)
		for (unsigned int y = frustum_box.m_min.y(); y < frustum_box.m_max.y(); y++)
			for (unsigned int x = frustum_box.m_min.x(); x < frustum_box.m_max.x(); x++)
			{
				// Transform from the cell world to the camera world
				Vector3f cell = rotation * m_volume->getWorldPosition(Vector3i(x, y, z)) + translation;

				// Project into a depth image
				cell = Transformations::reproject(cell, cloud->m_camera_parameters);

				// Pixels space
				auto pixels = round(cell);

				float depth = cloud->getDepthImage(pixels.x(), pixels.y());

				// Depth was not found
				if (depth == INFINITY) continue;

				m_mutex.lock();

				Voxel* voxel = m_volume->getVoxel(x, y, z);

				if (!voxel) continue;

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
					voxel->m_weight =
						min(int(weight) + int(m_weight_update), int(std::numeric_limits<unsigned char>::infinity()));
					voxel->m_position = Vector3f(x, y, z);
				}

				m_weight_update += weight;

				m_mutex.unlock();
			}

	SAFE_DELETE(cloud);
}

FrustumBox Fusion::computeFrustumBounds(Matrix4f cameraToWorld, CameraParameters camera_parameters) const{

	const auto rotation = cameraToWorld.block(0, 0, 3, 3);
	const auto translation = cameraToWorld.block(0, 3, 3, 1);

	auto width = camera_parameters.m_image_width;
	auto height = camera_parameters.m_image_height;
	auto min_depth = camera_parameters.m_depth_min;
	auto max_depth = camera_parameters.m_depth_max;

	std::vector<Vector3f> corners;

	// Image -> Camera -> World -> Grid
	for (auto depth : std::vector<float>{min_depth, max_depth})
	{
		corners.push_back(Transformations::backproject(0, 0, depth, camera_parameters));
		corners.push_back(Transformations::backproject(width - 1, 0, depth, camera_parameters));
		corners.push_back(Transformations::backproject(width - 1, height - 1, depth, camera_parameters));
		corners.push_back(Transformations::backproject(0, height - 1, depth, camera_parameters));
	}

	Vector3i min;
	Vector3i max;

	for (int i = 0; i < 8; i++)
	{
		auto grid = m_volume->getGridPosition(rotation * corners[i] + translation);

		if (grid.x() > max.x()) max[0] = grid[0];
		if (grid.y() > max.y()) max[1] = grid[1];
		if (grid.z() > max.z()) max[2] = grid[2];

		if (grid.x() < min.x()) min[0] = grid[0];
		if (grid.y() < min.y()) min[1] = grid[1];
		if (grid.z() < max.z()) min[2] = grid[2];
	}

	FrustumBox box;

	box.m_min = clamp(min);
	box.m_max = clamp(max);

	return box;
}
