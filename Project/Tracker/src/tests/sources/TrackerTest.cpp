#include "../headers/TrackerTest.h"

void TrackerTest::cameraPoseTest(){
	std::cout << "START cameraPoseTest()" << std::endl;

	TrackerContext* tracker_context = static_cast<TrackerContext*>(createContext(
		DatasetManager::getCurrentPath().data()));

	Matrix4f prev_trajectory;
	Matrix4f firs_trajectory_inverse = getTrajectory(0).inverse();

	//Statistics variables
	double icp_time = 0.0;
	float final_error = 0.0f, avg_error = 0.0f;
	float avg_displacement_error = 0.0f, displacement_error = 0.0f;

	int nIters = 700; //3000
	Matrix4f trajectory;

	// for some reason when the scope of this mat is inside the loop it gets borked after alignNewFrame() is called 
	for (int i = 0; i < nIters; i++)
	{
		trajectory = firs_trajectory_inverse * getTrajectory(i); //get camera trajectory of index from testBase class
		cv::Mat rgb, depth;

		dynamic_cast<DatasetVideoStreamReader*>(tracker_context->m_videoStreamReader)->readAnyFrame(i, rgb, depth);

		PointCloud* _source = new PointCloud(tracker_context->m_tracker->getCameraParameters(), depth, rgb, 8);
		std::shared_ptr<PointCloud> source(_source);

		if (i == 0) // first frame
		{
			tracker_context->m_tracker->m_pose = Matrix4f::Identity();
			tracker_context->m_tracker->m_previous_point_cloud = source;
			prev_trajectory = trajectory;
			continue;
		}

		//tracker_context->m_tracker->m_previous_point_cloud->transform(tracker_context->m_tracker->m_pose);

		//std::cout << "Previous Pose" << std::endl;
		//std::cout << tracker_context->m_tracker->m_pose << std::endl;

		std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

		Matrix4f deltaPose = tracker_context->m_tracker->alignNewFrame(
			source, tracker_context->m_tracker->m_previous_point_cloud);

		Matrix4f pose = deltaPose * tracker_context->m_tracker->m_pose;

		std::cout << "Vertices: source: " << source->getPoints().size() << " target: " << tracker_context
		                                                                                  ->m_tracker->
		                                                                                  m_previous_point_cloud->
		                                                                                  getPoints().size() << std::
			endl;

		std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
		icp_time += time_span.count();

		// Safe the last frame reference
		tracker_context->m_tracker->m_previous_point_cloud = source;
		tracker_context->m_tracker->m_pose = pose;

		//trackerCameraPose(pc, img, pose);
		imshow("dllTest", rgb);
		cv::waitKey(1);

		//Compute the inverse of the pose
		//matPose = matPose.inverse().eval();

		std::cout << "\n ------- pose: " << i << " -------- \n" << pose
			<< "\n------------------------ " << std::endl;

		std::cout << "\n ------- trajectory: " << i << " -------- \n" << trajectory
			<< "\n------------------------ \n" << std::endl;

		std::cout << "\n ------- Delta Pose: " << i << " -------- \n" << deltaPose
			<< "\n------------------------ \n" << std::endl;

		//std::cout << "\n ------- trajectory difference: " << i << " -------- \n" << prev_trajectory.inverse() * trajectory
		//	<< "\n------------------------ \n" << std::endl;

		prev_trajectory = trajectory;

		//Error using Frobenius norm
		//Performance metric should be Absolute Trajectory Error (ATE) https://vision.in.tum.de/data/datasets/rgbd-dataset/tools#evaluation

		Matrix4f error = pose - trajectory;

		double prev_drift = final_error;
		double prev_displacement_drift = displacement_error;

		final_error = error.norm();
		displacement_error = error.block(0, 3, 3, 1).norm();

		avg_error += fabs(final_error - prev_drift);
		avg_displacement_error += fabs(displacement_error - prev_displacement_drift);

		std::cout << "\n ------- Error: " << i << " -------- \n" << final_error
			<< "\n ------- Translational Error: " << i << " -------- \n" << displacement_error

			<< "\n------------------------ " << std::endl;
	}

	std::cout << "Average ICP time:  " << 1000.0 * icp_time / nIters << " ms\n";
	std::cout << "Total ICP error:  " << 100.0 * final_error << " cm\n";
	std::cout << "Average ICP error:  " << 100.0* avg_error / nIters << " cm\n";
	std::cout << "Total ICP displacement error:  " << 100.0 * displacement_error << " cm\n";
	std::cout << "Average ICP displacement error:  " << 100.0 * avg_displacement_error / nIters << " cm\n";

	std::cin.get();
}

void TrackerTest::processedMapsTest(){
	std::cout << "START processedMapsTest()" << std::endl;

	TrackerContext* tracker_context = static_cast<TrackerContext*>(createContext(
		DatasetManager::getCurrentPath().data()));

	int nIters = 3000; //3000

	for (int i = 0; i < nIters; i++)
	{
		cv::Mat rgb, depth;

		dynamic_cast<DatasetVideoStreamReader*>(tracker_context->m_videoStreamReader)->readAnyFrame(i, rgb, depth);

		PointCloud* source = new PointCloud(tracker_context->m_tracker->getCameraParameters(), depth, rgb);

		cv::Mat scaled_depth, render_depth;
		double min, max;

		minMaxIdx(depth, &min, &max);
		convertScaleAbs(depth, scaled_depth, 255 / max);
		imshow("Raw Depth", scaled_depth);

		//Bilateral filter
		cv::Mat bilateral_depth = source->filterMap(depth, bilateral, 9, 32.0f);

		minMaxIdx(bilateral_depth, &min, &max);
		convertScaleAbs(bilateral_depth, render_depth, 255 / max);

		//imshow("Bilateral Filtered Depth", render_depth);

		//Median Filter
		cv::Mat median_depth = source->filterMap(scaled_depth, median, 7, 150.0f);

		//imshow("Median Filtered Depth", median_depth);

		//Normal maps (Pending task)
		cv::Mat normal_map = source->getNormalMap();

		imshow("Normal Map", normal_map);

		cv::waitKey(10);

		//cin.get();
	}
}
