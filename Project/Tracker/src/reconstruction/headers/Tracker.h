#ifndef PROJECT_TRACKER_H
#define PROJECT_TRACKER_H

#include <iostream>
#include <cstddef>

#include "../../data-stream/headers/VideoStreamReaderBase.h"
#include "../../data-stream/headers/DatasetVideoStreamReader.h"
#include "icp/ICPComplete.h"
#include "icp/ICPNaive.h"

using namespace std;

enum ICPType {NAIVE, PCL};

/**
 * Tracks frame to frame transition and estimate the pose
 */
class Tracker final
{
public:
	Tracker(SystemParameters camera_parameters, ICPType icp_type) : m_camera_parameters(camera_parameters){

		switch (icp_type)
		{
			case NAIVE: m_icp = new ICPNaive(); break;
			case PCL: m_icp = new ICPComplete(); break;
		}
	}

	~Tracker();

	Matrix4f alignNewFrame(std::shared_ptr<PointCloud> sourcePoints, std::shared_ptr<PointCloud> targetPoints) const;

	SystemParameters getCameraParameters() const;

	std::shared_ptr<PointCloud> m_previous_point_cloud = nullptr;

	Matrix4f m_pose = Matrix4f::Identity();

private:
	ICP* m_icp = nullptr;

	SystemParameters m_camera_parameters;
};

#endif //PROJECT_TRACKER_H
