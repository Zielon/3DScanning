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
	Tracker(SystemParameters system_parameters, ICPType icp_type) : m_system_parameters(system_parameters){

		switch (icp_type)
		{
		case NAIVE: m_icp = new ICPNaive(system_parameters);
			break;
		case NON_LINEAR: m_icp = new ICPNonLinear(system_parameters);
			break;
		case FEATURES: m_icp = new ICPFeatures(system_parameters);
			break;
		case CUDA: m_icp = new ICPCUDA(system_parameters);
			break;
		}

		this->icp_type = icp_type;
	}

	~Tracker();

	Matrix4f alignNewFrame(std::shared_ptr<PointCloud> sourcePoints, std::shared_ptr<PointCloud> targetPoints) const;

	SystemParameters getSystemParameters() const;

	std::shared_ptr<PointCloud> m_previous_point_cloud = nullptr;

	Matrix4f m_pose = Matrix4f::Identity();

	ICP* m_icp = nullptr;

	ICPType icp_type;

	SystemParameters m_system_parameters;
};

#endif //PROJECT_TRACKER_H
