#include "../headers/TrackerTest.h"

void TrackerTest::cameraPoseTest(){
	std::cout << "START cameraPoseTest()" << std::endl;

	TrackerContext* tracker_context = static_cast<TrackerContext*>(createContext(
		DatasetManager::getCurrentPath().data()));

	Matrix4f prev_trajectory;

	int nIters = 50; //3000

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

		Matrix4f deltaPose = tracker_context->m_tracker->alignNewFrame(
			source, tracker_context->m_tracker->m_previous_point_cloud);
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

void TrackerTest::processedMapsTest()
{
	std::cout << "START processedMapsTest()" << std::endl;

	TrackerContext* tracker_context = static_cast<TrackerContext*>(createContext(
		DatasetManager::getCurrentPath().data()));


	int nIters = 3000; //3000

	for (int i = 0; i < nIters; i++)
	{
		cv::Mat rgb, depth;

		dynamic_cast<DatasetVideoStreamReader*>(tracker_context->m_videoStreamReader)->readAnyFrame(i, rgb, depth);

		PointCloud* source = new PointCloud(tracker_context->m_tracker->getCameraParameters(), depth, rgb);

		
		cv::Mat scaled_depth, renderdepth;
		double min, max;

		cv::minMaxIdx(depth, &min, &max);
		cv::convertScaleAbs(depth, scaled_depth, 255 / max);
		cv::imshow("Raw Depth", scaled_depth);

		//Bilateral filter
		cv::Mat render_depth;
		cv::Mat bilateral_depth = source->filterMap(depth, bilateral, 9, 150.0f);

		cv::minMaxIdx(bilateral_depth, &min, &max);
		cv::convertScaleAbs(bilateral_depth, render_depth, 255 / max);

		cv::imshow("Bilateral Filtered Depth", render_depth);

		//Median Filter
		cv::Mat median_depth = source->filterMap(scaled_depth, median, 9, 150.0f);

		cv::imshow("Median Filtered Depth", median_depth);

		cv::waitKey(10);
	}

	/*double min, max;
	cv::Mat scaled_depth, scale_depth2;
	int diameter = 9;
	float sigma = 150.0f;

	cv::bilateralFilter(depth_mat, filtered_depth, diameter, sigma, sigma);

	//Show raw depth map
	cv::minMaxIdx(depth_mat, &min, &max);
	cv::convertScaleAbs(depth_mat, scaled_depth, 255 / max);
	cv::imshow("Raw Depth", scaled_depth);

	//cv::medianBlur(scaled_depth, filtered_depth, diameter);

	//Show filtered depth map

	cv::minMaxIdx(filtered_depth, &min, &max);
	cv::convertScaleAbs(filtered_depth, scaled_depth, 255 / max);

	cv::imshow("Filtered Depth", scaled_depth);*/
}
