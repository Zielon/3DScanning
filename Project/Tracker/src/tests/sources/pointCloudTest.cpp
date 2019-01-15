#include "../headers/pointCloudTest.h"

pointCloudTest::pointCloudTest()
{
	Verbose::message("START streamPointCloudTest()");

	TrackerContext* context = static_cast<TrackerContext*>(createContext(DatasetManager::getCurrentPath().data()));

	unsigned char* img = new unsigned char[getImageWidth(context) * getImageHeight(context) * 3];

	for (int index = 0; index < 500; index += 100)
	{
		const auto trajectory = getTrajectory(index); //get camera trajectory of index
		cv::Mat rgb, depth;

		dynamic_cast<DatasetVideoStreamReader*>(context->m_videoStreamReader)->readAnyFrame(index, rgb, depth);
		PointCloud* cloud = new PointCloud(context->m_tracker->getCameraParameters(), depth, rgb, false);

		ThreadManager::add([cloud, index, trajectory]() {
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


pointCloudTest::~pointCloudTest()
{
}
