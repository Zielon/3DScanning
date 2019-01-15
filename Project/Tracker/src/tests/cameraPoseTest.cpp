#include "cameraPoseTest.h"



cameraPoseTest::cameraPoseTest()
{
	std::cout << "START cameraPoseTest()" << std::endl;

	TrackerContext* pc = static_cast<TrackerContext*>(createContext(DatasetManager::getCurrentPath().data()));

	unsigned char* img = new unsigned char[getImageWidth(pc) * getImageHeight(pc) * 3];

	float pose[16];

	//Read groundtruth trajectories (camera poses)
	std::vector<Matrix4f> trajectories;
	std::vector<double> trajectory_timestamps;
	std::vector<double> depth_timestamps;

	m_files_manager.readTrajectoryFile(trajectories, trajectory_timestamps);
	m_files_manager.readDepthTimeStampFile(depth_timestamps);


	Matrix4f first_traj = Matrix4f::Identity();
	for (int i = 0; i < 3000; ++i)
	{

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

		if (i == 0) first_traj = trajectories[idx];
		const auto trajectory = trajectories[idx];

		//dllMain(pc, img, pose);
		memcpy(pose, first_traj.data(), 16 * sizeof(float));
		dllMain(pc, img, pose);

		cv::Mat dllmat = cv::Mat(getImageHeight(pc), getImageWidth(pc), CV_8UC3, img);
		imshow("dllTest", dllmat);
		cv::waitKey(1);

		Matrix4f matPose = Map<Matrix4f>(pose, 4, 4);

		//Compute the inverse of the pose
		//matPose = matPose.inverse().eval();

		std::cout << "\n ------- pose: " << i << " -------- \n" << matPose
			<< "\n------------------------ " << std::endl;

		std::cout << "\n ------- trajectory: " << i << " -------- \n" << trajectory
			<< "\n------------------------ " << std::endl;

		//Error using Frobenius norm
		//Performance metric should be Absolute Trajectory Error (ATE) https://vision.in.tum.de/data/datasets/rgbd-dataset/tools#evaluation

		Matrix4f error = matPose - trajectory;

		std::cout << "\n ------- Error: " << i << " -------- \n" << error.norm()
			<< "\n------------------------ " << std::endl;

		std::cin.get();
	}
}


cameraPoseTest::~cameraPoseTest()
{
}
