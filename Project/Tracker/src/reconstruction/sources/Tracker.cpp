#include "../headers/Tracker.h"

Tracker::~Tracker(){
	SAFE_DELETE(m_icp);
}

Matrix4f Tracker::alignNewFrame(std::shared_ptr<PointCloud> source, std::shared_ptr<PointCloud> target) const{
	return m_icp->estimatePose(source, target);
}

CameraParameters Tracker::getCameraParameters() const{
	return m_camera_parameters;
}
