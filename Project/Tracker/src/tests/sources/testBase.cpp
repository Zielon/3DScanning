#include "../headers/TestBase.h"
#include "../headers/DataStreamTest.h"
#include "../headers/ReconstructionTest.h"
#include "../headers/TrackerTest.h"

TestBase::TestBase() {
	//Read groundtruth trajectories (camera poses)
	std::vector<Matrix4f> trajectories;
	std::vector<double> trajectory_timestamps;
	std::vector<double> depth_timestamps;
	if (!m_trajectories.empty())
		return;
	m_files_manager.readTrajectoryFile(trajectories, trajectory_timestamps);
	m_files_manager.readDepthTimeStampFile(depth_timestamps);
	for (int i = 0; i < 600; i++) {
		//Finding proper trajectory
		double timestamp = depth_timestamps[i];
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
		m_trajectories.push_back(trajectories[idx]);
	}

};
std::vector<Eigen::Matrix4f> TestBase::m_trajectories;