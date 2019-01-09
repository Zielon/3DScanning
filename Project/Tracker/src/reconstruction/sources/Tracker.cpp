#include "../headers/Tracker.h"

Tracker::~Tracker(){
	delete m_icp;
}

void Tracker::alignNewFrame(const PointCloud& source, const PointCloud& target, float* outPose){

	const auto pose = m_icp->estimatePose(source, target).data();

	for (int i = 0; i < 16; i++)
		outPose[i] = pose[i];

}

CameraParameters Tracker::getCameraParameters() const{
	return m_camera_parameters;
}
