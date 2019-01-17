#include "../headers/ReconstructionTest.h"
#include "../../debugger/headers/ProgressBar.hpp"

void ReconstructionTest::pointCloudTest() const{

	Verbose::message("START streamPointCloudTest()");

	TrackerContext* context = static_cast<TrackerContext*>(createContext(DatasetManager::getCurrentPath().data()));

	unsigned char* img = new unsigned char[getImageWidth(context) * getImageHeight(context) * 3];

	for (int index = 0; index < 500; index += 100)
	{
		const auto trajectory = getTrajectory(index); //get camera trajectory of index from testBase class
		cv::Mat rgb, depth;

		dynamic_cast<DatasetVideoStreamReader*>(context->m_videoStreamReader)->readAnyFrame(index, rgb, depth);
		PointCloud* cloud = new PointCloud(context->m_tracker->getCameraParameters(), depth, rgb, false);

		ThreadManager::add([cloud, index, trajectory](){
			cloud->m_mesh.transform(trajectory);
			cloud->m_mesh.save("point_cloud_" + std::to_string(index));
			delete cloud;
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

	unsigned char* img = new unsigned char[getImageWidth(context) * getImageHeight(context) * 3];

	auto size = getIterations();

	ProgressBar bar(size, 60, "Frames loaded");

	for (int index = 0; index < size; index += 3)
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

	delete[]img;
	SAFE_DELETE(context);
}
