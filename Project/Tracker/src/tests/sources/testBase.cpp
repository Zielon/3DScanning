#include "../headers/testBase.h"

const Eigen::Matrix4f testBase::getTrajectory(int index) { //get trajectory
	//Read groundtruth trajectories (camera poses)
	std::vector<Matrix4f> trajectories;
	std::vector<double> trajectory_timestamps;
	std::vector<double> depth_timestamps;

	m_files_manager.readTrajectoryFile(trajectories, trajectory_timestamps);
	m_files_manager.readDepthTimeStampFile(depth_timestamps);

	//Finding proper trajectory
	double timestamp = depth_timestamps[index];
	double min = std::numeric_limits<double>::infinity();
	int idx = 0;
	for (unsigned int j = 0; j < trajectories.size(); ++j)
	{
		double d = abs(trajectory_timestamps[j] - timestamp);
		if (min > d)
		{
			min = d;
			idx = j;
		}
	}
	return trajectories[idx];
}

