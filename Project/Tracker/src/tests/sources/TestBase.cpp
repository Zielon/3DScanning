#include "../headers/TestBase.h"
#include "../headers/DataStreamTest.h"
#include "../headers/ReconstructionTest.h"
#include "../headers/TrackerTest.h"

std::vector<Matrix4f> TestBase::m_trajectories;
std::vector<double> TestBase::m_trajectory_timestamps;
std::vector<double> TestBase::m_depth_timestamps;

TestBase::TestBase(){
	if (!m_trajectories.empty())
		return;

	m_files_manager.readTrajectoryFile(m_trajectories, m_trajectory_timestamps);
	m_files_manager.readDepthTimeStampFile(m_depth_timestamps);
}

Matrix4f TestBase::getTrajectory(int index) const{
	const double timestamp = m_depth_timestamps[index];
	double min = std::numeric_limits<double>::infinity();
	int idx = 0;
	for (unsigned int j = 0; j < m_trajectories.size(); ++j)
	{
		double d = abs(m_trajectory_timestamps[j] - timestamp);
		if (min > d)
		{
			min = d;
			idx = j;
		}
	}

	return m_trajectories[idx];
}

int TestBase::getIterations(){
	return m_depth_timestamps.size();
};
