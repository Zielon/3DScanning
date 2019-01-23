#include "../headers/Tracker.h"

Tracker::~Tracker(){
	SAFE_DELETE(m_icp);
}

Matrix4f Tracker::alignNewFrame(std::shared_ptr<PointCloud> model, std::shared_ptr<PointCloud> data) const{
	return m_icp->estimatePose(model, data);
}

CameraParameters Tracker::getCameraParameters() const{
	return m_camera_parameters;
}
