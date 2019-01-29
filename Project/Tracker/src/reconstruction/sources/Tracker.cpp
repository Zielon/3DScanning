#include "../headers/Tracker.h"

Tracker::~Tracker(){
	SAFE_DELETE(m_icp);
}

Matrix4f Tracker::alignNewFrame(std::shared_ptr<PointCloud> model, std::shared_ptr<PointCloud> data) const{
	
	if (icp_type == CUDA) {
		return m_icp->estimatePose(model, data);
	}

	return m_icp->estimatePose(model, data) * m_pose;
}

SystemParameters Tracker::getSystemParameters() const{
	return m_system_parameters;
}
