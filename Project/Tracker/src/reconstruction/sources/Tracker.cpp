#include "../headers/Tracker.h"

Tracker::~Tracker(){
	SAFE_DELETE(m_icp);
}

Matrix4f Tracker::alignNewFrame(std::shared_ptr<PointCloud> model, std::shared_ptr<PointCloud> data, Matrix4f previous_pose) const{
	return m_icp->estimatePose(model, data);
}

SystemParameters Tracker::getCameraParameters() const{
	return m_camera_parameters;
}
