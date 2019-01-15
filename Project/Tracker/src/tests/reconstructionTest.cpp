#include "reconstructionTest.h"



reconstructionTest::reconstructionTest()
{
	Verbose::message("START reconstructionTest()");

	TrackerContext* context = static_cast<TrackerContext*>(createContext(DatasetManager::getCurrentPath().data()));

	unsigned char* img = new unsigned char[getImageWidth(context) * getImageHeight(context) * 3];

	for (int index = 0; index < 600; index += 50)
	{
		const auto trajectory = getTrajectory(index); //get camera trajectory of index
		cv::Mat rgb, depth;

		dynamic_cast<DatasetVideoStreamReader*>(context->m_videoStreamReader)->readAnyFrame(index, rgb, depth);
		PointCloud* cloud = new PointCloud(context->m_tracker->getCameraParameters(), depth, rgb, true);
		cloud->m_pose_estimation = trajectory;

		context->m_fusion->produce(cloud);
	}

	while (!context->m_fusion->isFinished())
		std::this_thread::sleep_for(std::chrono::seconds(1));

	context->m_fusion->save("mesh");

	Verbose::message("DONE reconstructionTest()", SUCCESS);

	delete[]img;
	SAFE_DELETE(context);
}


reconstructionTest::~reconstructionTest()
{
}
