#include "../headers/Tracker.h"

Tracker::~Tracker(){
	SAFE_DELETE(m_icp);
}

Matrix4f Tracker::alignNewFrame(const PointCloud* source, const PointCloud* target, float* outPose){

	const auto pose = m_icp->estimatePose(source, target);

	const auto data = pose.data();

	for (int i = 0; i < 16; i++)
		outPose[i] = data[i];

	return pose;
}

CameraParameters Tracker::getCameraParameters() const{
	return m_camera_parameters;
}
