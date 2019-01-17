#include "../headers/ReconstructionTest.h"

void ReconstructionTest::meshTest()
{
	std::cout << "START meshTest()" << std::endl;

	TrackerContext* pc = static_cast<TrackerContext*>(createContext(DatasetManager::getCurrentPath().data()));

	unsigned char* img = new unsigned char[getImageWidth(pc) * getImageHeight(pc) * 3];

	float pose[16];

	_mkdir("test_meshes");

	for (int i = 0; i < 3000; ++i)
	{
		dllMain(pc, img, pose);

		cv::Mat dllmat = cv::Mat(getImageHeight(pc), getImageWidth(pc), CV_8UC3, img);
		imshow("dllTest", dllmat);
		cv::waitKey(1);

		Matrix4f matPose = Map<Matrix4f>(pose, 4, 4);

		std::string filename = "test_meshes\\meshTest";
		filename += std::to_string(pc->m_videoStreamReader->getCurrentFrameIndex());
		filename += ".off";
		std::ofstream outFile(filename);
		if (!outFile.is_open()) continue;

		// write header
		outFile << "COFF" << std::endl;
		outFile << "# numVertices numFaces numEdges" << std::endl;
		outFile << getVertexCount(pc) << " ";
		assert(getIndexCount(pc) % 3 == 0);
		outFile << getIndexCount(pc) / 3 << " 0" << std::endl;

		outFile << "# list of vertices\n# X Y Z R G B A" << std::endl;

		float* vertexBuffer = new float[3 * getVertexCount(pc)];
		getVertexBuffer(pc, vertexBuffer);
		for (size_t i = 0; i < getVertexCount(pc); ++i)
		{
			//std::cout << vertexBuffer[3 * i + 0] << " "
			//	<< vertexBuffer[3 * i + 1] << " "
			//	<< vertexBuffer[3 * i + 2] << std::endl; 

			outFile << vertexBuffer[3 * i + 0] << " "
				<< vertexBuffer[3 * i + 1] << " "
				<< vertexBuffer[3 * i + 2] << " "
				<< (int)255 << " "
				<< (int)255 << " "
				<< (int)255 << " "
				<< (int)255 << std::endl;
		}

		outFile << "# list of faces\n# nVerticesPerFace idx0 idx1 idx2 ..." << std::endl;
		int* indexbuffer = new int[getIndexCount(pc)];
		getIndexBuffer(pc, indexbuffer);
		for (size_t i = 0; i < getIndexCount(pc) / 3; ++i)
		{
			//std::cout << vertexBuffer[3 * i + 0] << " "
			//	<< vertexBuffer[3 * i + 1] << " "
			//	<< vertexBuffer[3 * i + 2] << std::endl; 

			outFile << "3 " <<
				indexbuffer[3 * i + 0] << " "
				<< indexbuffer[3 * i + 1] << " "
				<< indexbuffer[3 * i + 2] << std::endl;
		}

		outFile.flush();
		outFile.close();

		std::cout << "\n ------- pose: " << i << " -------- \n" << matPose
			<< "\n------------------------ " << std::endl;
	}

	delete[]img;
	SAFE_DELETE(pc);
}

void ReconstructionTest::pointCloudTest()
{
	Verbose::message("START streamPointCloudTest()");

	TrackerContext* context = static_cast<TrackerContext*>(createContext(DatasetManager::getCurrentPath().data()));

	unsigned char* img = new unsigned char[getImageWidth(context) * getImageHeight(context) * 3];

	for (int index = 0; index < 500; index += 100)
	{
		const auto trajectory = m_trajectories[index]; //get camera trajectory of index from testBase class
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


void ReconstructionTest::reconstructTest()
{
	Verbose::message("START reconstructionTest()");

	TrackerContext* context = static_cast<TrackerContext*>(createContext(DatasetManager::getCurrentPath().data()));

	unsigned char* img = new unsigned char[getImageWidth(context) * getImageHeight(context) * 3];

	for (int index = 0; index < 600; index += 50)
	{
		const auto trajectory = m_trajectories[index]; //get camera trajectory of index from testBase class
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

void ReconstructionTest::pointCloudWithIcpTest()
{
	Verbose::message("START streamPointCloudWithICPTest()");

	TrackerContext* tracker_context = static_cast<TrackerContext*>(createContext(DatasetManager::getCurrentPath().data()));

	unsigned char* img = new unsigned char[getImageWidth(tracker_context) * getImageHeight(tracker_context) * 3];

	int startFrame = 0;
	Matrix4f pose;

	for (int index = startFrame; index < 40; index += 10)
	{
		const auto trajectory = m_trajectories[index]; //get camera trajectory of index from testBase class
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

		ThreadManager::add([source, index, pose]() {
			source->m_mesh.transform(pose);
			source->m_mesh.save("point_cloud_" + std::to_string(index));
			delete source;
		});
	}

	ThreadManager::waitForAll();

	Verbose::message("DONE streamPointCloudTest()", SUCCESS);

	delete[]img;
	SAFE_DELETE(tracker_context);

}
