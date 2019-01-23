#ifndef TRACKER_LIB_FUSION_H
#define TRACKER_LIB_FUSION_H

#include <iostream>

#include "../../Eigen.h"
#include "CameraParameters.h"
#include "Voxel.h"
#include "PointCloud.h"
#include "Volume.h"
#include "../../concurency/sources/Buffer.cpp"
#include "../../concurency/sources/Consumer.cpp"

using namespace std;

struct FrustumBox
{
	Vector3i m_max;
	Vector3i m_min;
};

/**
 * Volumetric m_fusion class
 */
class Fusion final
{
public:
	Fusion(CameraParameters camera_parameters, int downSamplingFactor = 2);

	~Fusion();

	void consume();

	void produce(std::shared_ptr<PointCloud> cloud) const;

	void integrate(std::shared_ptr<PointCloud> cloud) const;

	void save(string name) const;

	void processMesh(Mesh& mesh) const;

	void wait() const;

private:
	void initialize();

	FrustumBox computeFrustumBounds(Matrix4f cameraToWorld, CameraParameters camera_parameters) const;

	int clamp(float value) const;

	Vector3i clamp(Vector3i value) const;

	void stopConsumers();

	float getWeight(float depth, float max) const;

	int m_downsampling_factor; 
	float m_trunaction = 0;
	std::vector<std::thread> m_consumer_threads;
	std::vector<Consumer<std::shared_ptr<PointCloud>>*> m_consumers;
	Buffer<std::shared_ptr<PointCloud>>* m_buffer;
	Volume* m_volume;
	std::mutex m_mutex;
	CameraParameters m_camera_parameters;
	const int NUMBER_OF_CONSUMERS = 5;
};

#endif //TRACKER_LIB_FUSION_H
