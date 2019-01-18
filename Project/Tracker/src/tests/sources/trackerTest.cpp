#include "../headers/TrackerTest.h"

void TrackerTest::cameraPoseTest(){
	std::cout << "START cameraPoseTest()" << std::endl;

	TrackerContext* tracker_context = static_cast<TrackerContext*>(createContext(
		DatasetManager::getCurrentPath().data()));

	Matrix4f prev_trajectory;
	Matrix4f firs_trajectory_inverse = getTrajectory(0).inverse();

	int nIters = 50; //3000
	Matrix4f trajectory; // for some reason when the scope of this mat is inside the loop it gets borked after alignNewFrame() is called 
	for (int i = 0; i < nIters; i++)
	{
		trajectory = firs_trajectory_inverse * getTrajectory(i); //get camera trajectory of index from testBase class
		cv::Mat rgb, depth;

		dynamic_cast<DatasetVideoStreamReader*>(tracker_context->m_videoStreamReader)->readAnyFrame(i, rgb, depth);

		PointCloud* source = new PointCloud(tracker_context->m_tracker->getCameraParameters(), depth, rgb, 8 );

		if (i == 0) // first frame
		{
			tracker_context->m_tracker->m_previous_pose = Matrix4f::Identity();
			tracker_context->m_tracker->m_previous_point_cloud = source;
			prev_trajectory = trajectory; 
			continue;
		}

		//tracker_context->m_tracker->m_previous_point_cloud->transform(tracker_context->m_tracker->m_previous_pose);

		//std::cout << "Previous Pose" << std::endl;
		//std::cout << tracker_context->m_tracker->m_previous_pose << std::endl;

		Matrix4f deltaPose = tracker_context->m_tracker->alignNewFrame(
			source, tracker_context->m_tracker->m_previous_point_cloud);

		Matrix4f pose = deltaPose * tracker_context->m_tracker->m_previous_pose;

		std::cout << "Vertices: source: " << source->getPoints().size() << " target: " << tracker_context->m_tracker->m_previous_point_cloud->getPoints().size() << std::endl; 

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
			<< "\n------------------------ \n" << std::endl;

		std::cout << "\n ------- Delta Pose: " << i << " -------- \n" << deltaPose.inverse()
			<< "\n------------------------ \n" << std::endl;

		//std::cout << "\n ------- trajectory difference: " << i << " -------- \n" << prev_trajectory.inverse() * trajectory
		//	<< "\n------------------------ \n" << std::endl;

		prev_trajectory = trajectory;

		//Error using Frobenius norm
		//Performance metric should be Absolute Trajectory Error (ATE) https://vision.in.tum.de/data/datasets/rgbd-dataset/tools#evaluation

		Matrix4f error = pose - trajectory;

		std::cout << "\n ------- Error: " << i << " -------- \n" << error.norm()
			<< "\n ------- Translational Error: " << i << " -------- \n" << error.block(0, 3, 3, 1).norm()

			<< "\n------------------------ " << std::endl;


		//std::cin.get();
	}
}
