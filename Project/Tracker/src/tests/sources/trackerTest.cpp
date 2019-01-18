#include "../headers/TrackerTest.h"

void TrackerTest::cameraPoseTest() {
	std::cout << "START cameraPoseTest()" << std::endl;

	TrackerContext* tracker_context = static_cast<TrackerContext*>(createContext(DatasetManager::getCurrentPath().data()));

	Matrix4f prev_trajectory;

	int nIters = 50;//3000

	for (int i = 0; i < nIters; i++)
	{
		const auto trajectory = m_trajectories[i]; //get camera trajectory of index from testBase class

		cv::Mat rgb, depth;

		dynamic_cast<DatasetVideoStreamReader*>(tracker_context->m_videoStreamReader)->readAnyFrame(i, rgb, depth);

		PointCloud* source = new PointCloud(tracker_context->m_tracker->getCameraParameters(), depth, rgb);

		if (i == 0) // first frame
		{
			tracker_context->m_tracker->m_previous_pose = Matrix4f::Identity();
			tracker_context->m_tracker->m_previous_point_cloud = source;

			continue;
		}


		source->transform(tracker_context->m_tracker->m_previous_pose);
		//tracker_context->m_tracker->m_previous_point_cloud->transform(tracker_context->m_tracker->m_previous_pose);

		std::cout << "Previous Pose" << std::endl;
		std::cout << tracker_context->m_tracker->m_previous_pose << std::endl;

		Matrix4f deltaPose = tracker_context->m_tracker->alignNewFrame(source, tracker_context->m_tracker->m_previous_point_cloud);
		//Matrix4f deltaPose = tracker_context->m_tracker->alignNewFrame(tracker_context->m_tracker->m_previous_point_cloud, source);

		//Matrix4f pose = deltaPose * tracker_context->m_tracker->m_previous_pose;
		Matrix4f pose = tracker_context->m_tracker->m_previous_pose * deltaPose;

		std::cout << "Delta Pose" << std::endl;
		std::cout << deltaPose << std::endl;

		// Safe the last frame reference
		tracker_context->m_tracker->m_previous_point_cloud = source;
		tracker_context->m_tracker->m_previous_pose = pose;

		//trackerCameraPose(pc, img, pose);
		imshow("dllTest", rgb);
		cv::waitKey(1);

		//Compute the inverse of the pose
		//matPose = matPose.inverse().eval();

		std::cout << "\n ------- pose: " << i << " -------- \n" << pose
			<< "\n------------------------ " << std::endl;

		std::cout << "\n ------- trajectory: " << i << " -------- \n" << trajectory
			<< "\n------------------------ " << std::endl;

		std::cout << "\n ------- trajectory difference: " << i << " -------- \n" << trajectory - prev_trajectory
			<< "\n------------------------ " << std::endl;

		prev_trajectory = trajectory;

		//Error using Frobenius norm
		//Performance metric should be Absolute Trajectory Error (ATE) https://vision.in.tum.de/data/datasets/rgbd-dataset/tools#evaluation

		Matrix4f error = pose - trajectory;

		std::cout << "\n ------- Error: " << i << " -------- \n" << error.norm()
			<< "\n------------------------ " << std::endl;

		tracker_context->m_tracker->m_previous_point_cloud = source;
		tracker_context->m_tracker->m_previous_pose = pose;

		//std::cin.get();
	}
}

void TrackerTest::frameDistanceTest() {

	std::cout << "START frameDiffTest()" << std::endl;
	
	std::vector<cv::Mat> images;

	TrackerContext* tracker_context = static_cast<TrackerContext*>(createContext(DatasetManager::getCurrentPath().data()));

	Matrix4f prev_trajectory,pose;
	prev_trajectory = Matrix4f::Zero();

	int nIters = 180;//3000
	int startFrame = 150;

	for (int i = startFrame; i < nIters; i+=3)
	{
		cv::Mat rgb, depth;

		dynamic_cast<DatasetVideoStreamReader*>(tracker_context->m_videoStreamReader)->readAnyFrame(i, rgb, depth);

		images.push_back(rgb);

		PointCloud* source = new PointCloud(tracker_context->m_tracker->getCameraParameters(), depth, rgb);
		
		if (i == startFrame) // first frame
		{
			tracker_context->m_tracker->m_previous_pose = Matrix4f::Identity();
			tracker_context->m_tracker->m_previous_point_cloud = source;

			continue;
		}


		source->transform(tracker_context->m_tracker->m_previous_pose);
		//tracker_context->m_tracker->m_previous_point_cloud->transform(tracker_context->m_tracker->m_previous_pose);

		/*
		std::cout << "Previous Pose" << std::endl;
		std::cout << tracker_context->m_tracker->m_previous_pose << std::endl;
		*/

		Matrix4f deltaPose = tracker_context->m_tracker->alignNewFrame(source, tracker_context->m_tracker->m_previous_point_cloud);
		
		pose = deltaPose * tracker_context->m_tracker->m_previous_pose;
		
		/*
		std::cout << "Delta Pose" << std::endl;
		std::cout << deltaPose << std::endl;
		*/

		std::cout << "\n ------- pose: " << i << " -------- \n" << pose
			<< "\n------------------------ " << std::endl;

		std::cout << "\n ------- trajectory: " << i << " -------- \n" << m_trajectories[i]
			<< "\n------------------------ " << std::endl;

		std::cout << "\n ------- trajectory difference: " << i << " -------- \n" << m_trajectories[i] - prev_trajectory
			<< "\n------------------------ " << std::endl;

		std::cout << "\n ------- pose difference: " << i << " -------- \n" << pose - tracker_context->m_tracker->m_previous_pose
			<< "\n------------------------ " << std::endl;

		prev_trajectory = m_trajectories[i];
		// Save the last frame reference
		tracker_context->m_tracker->m_previous_point_cloud = source;
		tracker_context->m_tracker->m_previous_pose = pose;


		imshow("test",rgb);
		cv::waitKey(1);
	}	
	const Eigen::VectorXf frame1Pose = Matrix4f::Identity().col(3);
	const Eigen::VectorXf frame2Pose = pose.col(3);
	float icpDistance = (frame2Pose.head(3) - frame1Pose.head(3)).norm();

	const Eigen::VectorXf frame1Truth = m_trajectories[startFrame].col(3);
	const Eigen::VectorXf frame2Truth = m_trajectories[nIters - 1].col(3);
	float trajectoryDistance = (frame2Truth.head(3) - frame1Truth.head(3)).norm();

	/*
	std::cout << "frame1 translation " << frame1Truth.head(3) << endl;
	std::cout << "frame2 translation " << frame2Truth.head(3) << endl;
	*/
	std::cout << "Trajectory distance difference between frames: " << trajectoryDistance << endl; //Groundtruth distance btw two frames
	std::cout << "------------------------------------------------" << endl;
	
	/*
	std::cout << "frame1pose translation " << frame1Pose.head(3) << endl;
	std::cout << "frame2pose translation " << frame2Pose.head(3) << endl;
	*/

	std::cout << "ICP calculated distance difference between frames: " << icpDistance << endl; //ICP calculated pose distance btw two frames
	std::cout << "------------------------------------------------" << endl;
	std::cout << "Difference btw real and ICP: " << abs(icpDistance - trajectoryDistance) << endl; //ICP calculated pose distance btw two frames

}