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

/**
 * Volumetric m_fusion class
 */
class Fusion final
{
public:
	Fusion(CameraParameters camera_parameters);

	Fusion(int width, int height, int pixelSteps) : m_height(height), m_width(width), m_pixelSteps(pixelSteps){
		m_volume = new Volume(Vector3d(-0.1, -0.1, -0.1), Vector3d(1.1, 1.1, 1.1), m_volume_size, 1);
		m_buffer = new Buffer<PointCloud*>();
		m_consumer = new Consumer<PointCloud*>(m_buffer);
	}

	~Fusion();

	void consume();

	void produce(PointCloud* cloud) const;

	void integrate(PointCloud* cloud) const;

	std::vector<int> m_currentIndexBuffer;

	Volume* getVolume() const{
		return m_volume;
	}

	int m_volume_size = 25;

private:
	Buffer<PointCloud*>* m_buffer;
	Consumer<PointCloud*>* m_consumer;
	std::thread m_consumer_thread;
	CameraParameters m_camera_parameters;
	Volume* m_volume;

	int m_height, m_width, m_pixelSteps;
};

#endif //TRACKER_LIB_FUSION_H
