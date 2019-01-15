#include "../headers/trackerTest.h"



void trackerTest::cameraPoseTest(){
	std::cout << "START cameraPoseTest()" << std::endl;

	TrackerContext* tracker_context = static_cast<TrackerContext*>(createContext(DatasetManager::getCurrentPath().data()));

	Matrix4f prev_trajectory;

	int nIters = 50;//3000

	for (int i = 0; i < nIters; i++)
	{
		const auto trajectory = getTrajectory(i); //get camera trajectory of index from testBase class

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

		Matrix4f pose = deltaPose * tracker_context->m_tracker->m_previous_pose;

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