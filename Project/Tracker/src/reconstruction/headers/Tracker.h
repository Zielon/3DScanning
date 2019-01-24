#ifndef PROJECT_TRACKER_H
#define PROJECT_TRACKER_H

#include <iostream>
#include <cstddef>

#include "../../data-stream/headers/VideoStreamReaderBase.h"
#include "../../data-stream/headers/DatasetVideoStreamReader.h"
#include "icp/ICPComplete.h"

using namespace std;

/**
 * Tracks frame to frame transition and estimate the pose
 */
class Tracker final
{
public:
	Tracker(CameraParameters camera_parameters) : m_camera_parameters(camera_parameters) {
		//m_icp = new ICPNaive();
		m_icp = new ICPComplete();
	}

	~Tracker();

	Matrix4f alignNewFrame(std::shared_ptr<PointCloud> sourcePoints, std::shared_ptr<PointCloud> targetPoints) const;

	CameraParameters getCameraParameters() const;

	std::shared_ptr<PointCloud> m_previous_point_cloud = nullptr;

	Matrix4f m_previous_pose = Matrix4f::Identity();

private:
	ICP* m_icp = nullptr;

	CameraParameters m_camera_parameters;
};

#endif //PROJECT_TRACKER_H