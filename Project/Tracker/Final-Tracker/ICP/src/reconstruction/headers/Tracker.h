#ifndef PROJECT_TRACKER_H
#define PROJECT_TRACKER_H

#include <iostream>
#include <cstddef>

#include "../../data-stream/headers/VideoStreamReaderBase.h"
#include "../../data-stream/headers/DatasetVideoStreamReader.h"
#include "icp/ICPNonLinear.h"
#include "icp/ICPNaive.h"
#include "icp/ICPFeatures.h"
#include "icp/ICPCUDA.h"

using namespace std;

enum ICPType { NAIVE, NON_LINEAR, FEATURES, CUDA };

/**
 * Tracks frame to frame transition and estimate the pose
 */
class Tracker final
{
public:
	Tracker(SystemParameters camera_parameters, ICPType icp_type) : m_camera_parameters(camera_parameters) {

		switch (icp_type)
		{
		case NAIVE: m_icp = new ICPNaive();
			break;
		case NON_LINEAR: m_icp = new ICPNonLinear();
			break;
		case FEATURES: m_icp = new ICPFeatures();
			break;
		case CUDA: m_icp = new ICPCUDA();
			break;
		}
	}

	~Tracker();

	Matrix4f alignNewFrame(std::shared_ptr<PointCloud> model, std::shared_ptr<PointCloud> data) const;

	SystemParameters getCameraParameters() const;

	std::shared_ptr<PointCloud> m_previous_point_cloud = nullptr;

	Matrix4f m_pose = Matrix4f::Identity();

private:
	ICP* m_icp = nullptr;

	SystemParameters m_camera_parameters;
};

#endif //PROJECT_TRACKER_H