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
	Fusion(CameraParameters camera_parameters);

	~Fusion();

	void consume();

	void produce(PointCloud* cloud) const;

	void integrate(PointCloud* cloud);

	void save(string name) const;

	void wait() const;

	std::vector<int> m_currentIndexBuffer;

private:
	void initialize();

	Vector3f reproject(Vector3f point) const;

	FrustumBox computeFrustumBounds(Matrix4f pose, CameraParameters camera_parameters) const;

	int clamp(float value) const;

	Vector3i clamp(Vector3i value) const;

	void stopConsumers();

	Vector3f backproject(float x, float y, float depth) const;

	float m_weight_update = 1;
	std::vector<std::thread> m_consumer_threads;
	std::vector<Consumer<PointCloud*>*> m_consumers;
	Buffer<PointCloud*>* m_buffer;
	Volume* m_volume;
	std::mutex m_mutex;
	CameraParameters m_camera_parameters;
	const int NUMBER_OF_CONSUMERS = 3;
};

#endif //TRACKER_LIB_FUSION_H
