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
	bool m_is_valid;
	int m_min_x, m_max_x;
	int m_min_y, m_max_y;
	int m_min_z, m_max_z;
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

	void save(string name);

	std::vector<int> m_currentIndexBuffer;

private:
	void initialize();

	Vector3f reproject(Vector3f point) const;

	FrustumBox computeFrustumBounds(Matrix4f pose) const;
	float m_weight_update = 1;
	std::thread m_consumer_thread;
	Consumer<PointCloud*>* m_consumer;
	Buffer<PointCloud*>* m_buffer;
	CameraParameters m_camera_parameters;
	Volume* m_volume;
};

#endif //TRACKER_LIB_FUSION_H
