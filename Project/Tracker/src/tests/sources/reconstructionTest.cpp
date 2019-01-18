#include "../headers/ReconstructionTest.h"
#include "../../debugger/headers/ProgressBar.hpp"

void ReconstructionTest::pointCloudTest() const{

	Verbose::message("START streamPointCloudTest()");

	TrackerContext* context = static_cast<TrackerContext*>(createContext(DatasetManager::getCurrentPath().data()));

	unsigned char* img = new unsigned char[getImageWidth(context) * getImageHeight(context) * 3];

	for (int index = 0; index < 600; index += 100)
	{
		const auto trajectory = getTrajectory(index);

		// Process each point cloud in a different thread
		ThreadManager::add([context, index, trajectory](){
			cv::Mat rgb, depth;
			dynamic_cast<DatasetVideoStreamReader*>(context->m_videoStreamReader)->readAnyFrame(index, rgb, depth);
			Mesh mesh(depth, rgb, context->m_tracker->getCameraParameters());
			mesh.transform(trajectory);
			mesh.save("point_cloud_" + std::to_string(index));
		});
	}

	ThreadManager::waitForAll();

	Verbose::message("DONE streamPointCloudTest()", SUCCESS);

	delete[]img;
	SAFE_DELETE(context);

}

void ReconstructionTest::reconstructionTest() const{

	Verbose::message("START reconstructionTest()");

	TrackerContext* context = static_cast<TrackerContext*>(createContext(DatasetManager::getCurrentPath().data()));

	//unsigned char* img = new unsigned char[getImageWidth(context) * getImageHeight(context) * 3];

	auto size = getIterations();

	ProgressBar bar(size, 60, "Frames loaded");

	for (int index = 0; index < size; index += 5)
	{
		const auto trajectory = getTrajectory(index);
		cv::Mat rgb, depth;

		dynamic_cast<DatasetVideoStreamReader*>(context->m_videoStreamReader)->readAnyFrame(index, rgb, depth);
		PointCloud* cloud = new PointCloud(context->m_tracker->getCameraParameters(), depth, rgb, false);
		cloud->m_pose_estimation = trajectory;
		context->m_fusion->produce(cloud);

		bar.set(index);
		bar.display();
	}

	bar.done();

	context->m_fusion->save("mesh");

	Verbose::message("DONE reconstructionTest()", SUCCESS);

	//delete[]img;
	SAFE_DELETE(context);
}

void ReconstructionTest::pointCloudWithIcpTest()
{
	Verbose::message("START streamPointCloudWithICPTest()");

	TrackerContext* tracker_context = static_cast<TrackerContext*>(createContext(DatasetManager::getCurrentPath().data()));

	unsigned char* img = new unsigned char[getImageWidth(tracker_context) * getImageHeight(tracker_context) * 3];

	int startFrame = 0;
	Matrix4f pose;
	auto size = getIterations();

	ProgressBar bar(size, 60, "Frames loaded");

	for (int index = startFrame; index < size; index += 10)
	{
		const auto trajectory = getTrajectory(index); //get camera trajectory of index from testBase class
		cv::Mat rgb, depth;

		dynamic_cast<DatasetVideoStreamReader*>(tracker_context->m_videoStreamReader)->readAnyFrame(index, rgb, depth);
		PointCloud* source = new PointCloud(tracker_context->m_tracker->getCameraParameters(), depth, rgb, false);
		
		if (index == startFrame) // first frame
		{
			tracker_context->m_tracker->m_previous_pose = Matrix4f::Identity();
			tracker_context->m_tracker->m_previous_point_cloud = source;

			continue;
		}

		Matrix4f deltaPose = tracker_context->m_tracker->alignNewFrame(source, tracker_context->m_tracker->m_previous_point_cloud);
		pose = deltaPose * tracker_context->m_tracker->m_previous_pose;

		source->m_pose_estimation = pose;
		tracker_context->m_fusion->produce(source);

		bar.set(index);
		bar.display();
	}

	tracker_context->m_fusion->save("mesh");

	Verbose::message("DONE pointCloudWithIcpTest()", SUCCESS);


	delete[]img;
	SAFE_DELETE(tracker_context);

}
