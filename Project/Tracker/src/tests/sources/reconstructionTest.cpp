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

	unsigned char* img = new unsigned char[getImageWidth(context) * getImageHeight(context) * 3];

	auto size = getIterations();

	ProgressBar bar(size, 60, "Frames loaded");

	for (int index = 0; index < size; index += 5)
	{
		const auto trajectory = getTrajectory(index);
		cv::Mat rgb, depth;

		dynamic_cast<DatasetVideoStreamReader*>(context->m_videoStreamReader)->readAnyFrame(index, rgb, depth);
		PointCloud* cloud = new PointCloud(context->m_tracker->getCameraParameters(), depth, rgb, 1);
		cloud->m_pose_estimation = trajectory;
		context->m_fusion->produce(cloud);


		std::vector<Vector3f> dump; 
		dump.push_back(Vector3f::Zero()); 
		cloud->queryNearestNeighbor(dump); 
		--cloud->refCounter; //this would be decreased once by alignToFrame()
		if (!cloud->refCounter)
		{
			SAFE_DELETE(cloud); 
		}


		bar.set(index);
		bar.display();
	}

	bar.done();

	context->m_fusion->save("mesh");

	Verbose::message("DONE reconstructionTest()", SUCCESS);

	delete[]img;
	SAFE_DELETE(context);
}


void ReconstructionTest::pointCloudTestWithICP() const {

	Verbose::message("START pointCloudTestWithICP()");

	TrackerContext* context = static_cast<TrackerContext*>(createContext(DatasetManager::getCurrentPath().data()));
	
	int startFrame = 100;
	
	for (int index = startFrame; index < 120; index++)
	{
		cv::Mat rgb, depth;

		dynamic_cast<DatasetVideoStreamReader*>(context->m_videoStreamReader)->readAnyFrame(index, rgb, depth);

		PointCloud* source = new PointCloud(context->m_tracker->getCameraParameters(), depth, rgb, 8);

		if (index == startFrame) // first frame
		{
			context->m_tracker->m_previous_pose = Matrix4f::Identity();
			context->m_tracker->m_previous_point_cloud = source;

			continue;
		}

		Matrix4f deltaPose = context->m_tracker->alignNewFrame(source, context->m_tracker->m_previous_point_cloud);

		Matrix4f pose = deltaPose * context->m_tracker->m_previous_pose;

		context->m_tracker->m_previous_point_cloud = source;
		context->m_tracker->m_previous_pose = pose;

		Mesh mesh(depth, rgb, context->m_tracker->getCameraParameters());
		mesh.transform(pose);
		mesh.save("point_cloud_" + std::to_string(index));

	}

	Verbose::message("DONE pointCloudTestWithICP()", SUCCESS);
	SAFE_DELETE(context);

}