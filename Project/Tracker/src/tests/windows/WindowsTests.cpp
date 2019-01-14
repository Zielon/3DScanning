#include "WindowsTests.h"
#include "../../TrackerContext.h"

#include <fstream>
#include <sstream>
#include <direct.h>
#include <io.h>
#include "../../debugger/headers/Verbose.h"
#include "../../reconstruction/headers/Mesh.h"
#include "../../concurency/headers/ThreadManager.h"

void WindowsTests::run(){

	//reconstructionTest();
	//streamPointCloudTest();
	// meshTest();
	// vidReadTest();
	 cameraPoseTest();
}

void WindowsTests::meshTest(){

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

	delete []img;
	SAFE_DELETE(pc);
}

void WindowsTests::streamPointCloudTest() const{

	Verbose::message("START streamPointCloudTest()");

	TrackerContext* context = static_cast<TrackerContext*>(createContext(DatasetManager::getCurrentPath().data()));

	unsigned char* img = new unsigned char[getImageWidth(context) * getImageHeight(context) * 3];

	std::vector<Matrix4f> trajectories;
	std::vector<double> trajectory_timestamps;
	std::vector<double> depth_timestamps;

	m_files_manager.readTrajectoryFile(trajectories, trajectory_timestamps);
	m_files_manager.readDepthTimeStampFile(depth_timestamps);

	for (int index = 0; index < 500; index += 100)
	{
		//Finding proper trajectory
		double timestamp = depth_timestamps[index];
		double min = std::numeric_limits<double>::infinity();
		int idx = 0;
		for (unsigned int j = 0; j < trajectories.size(); ++j)
		{
			double d = abs(trajectory_timestamps[j] - timestamp);
			if (min > d)
			{
				min = d;
				idx = j;
			}
		}

		const auto trajectory = trajectories[idx];
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

void WindowsTests::reconstructionTest() const{

	Verbose::message("START reconstructionTest()");

	TrackerContext* context = static_cast<TrackerContext*>(createContext(DatasetManager::getCurrentPath().data()));

	unsigned char* img = new unsigned char[getImageWidth(context) * getImageHeight(context) * 3];

	std::vector<Matrix4f> trajectories;
	std::vector<double> trajectory_timestamps;
	std::vector<double> depth_timestamps;

	m_files_manager.readTrajectoryFile(trajectories, trajectory_timestamps);
	m_files_manager.readDepthTimeStampFile(depth_timestamps);

	for (int index = 0; index < 600; index += 50)
	{
		double timestamp = depth_timestamps[index];
		double min = std::numeric_limits<double>::infinity();
		int idx = 0;
		for (unsigned int j = 0; j < trajectories.size(); ++j)
		{
			double d = abs(trajectory_timestamps[j] - timestamp);
			if (min > d)
			{
				min = d;
				idx = j;
			}
		}

		const auto trajectory = trajectories[idx];
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

void WindowsTests::vidReadTest(){

	std::cout << "START vidReadTest()" << std::endl;

	VideoStreamReaderBase* videoInputReader = new DatasetVideoStreamReader(
		DatasetManager::getCurrentPath().data(), false);

	if (!videoInputReader->startReading())
	{
		std::cout << "Failed to read input stream" << std::endl;
		exit(-1);
	}

	cv::Mat rgb;
	cv::Mat depth;

	for (int i = 0; i < 3000; ++i)
	{
		videoInputReader->getNextFrame(rgb, depth, true);

		imshow("TestRGB", rgb);
		double min;
		double max;
		minMaxIdx(depth, &min, &max);
		cv::Mat adjMap;
		convertScaleAbs(depth, adjMap, 255 / max);
		imshow("TestDepth", adjMap);

		cv::waitKey(1);
	}
}

void WindowsTests::cameraPoseTest(){

	std::cout << "START cameraPoseTest()" << std::endl;

	TrackerContext* pc = static_cast<TrackerContext*>(createContext(DatasetManager::getCurrentPath().data()));

	unsigned char* img = new unsigned char[getImageWidth(pc) * getImageHeight(pc) * 3];

	float pose[16];

	//Read groundtruth trajectories (camera poses)
	std::vector<Matrix4f> trajectories;
	std::vector<double> trajectory_timestamps;
	std::vector<double> depth_timestamps;

	m_files_manager.readTrajectoryFile(trajectories, trajectory_timestamps);
	m_files_manager.readDepthTimeStampFile(depth_timestamps);


	Matrix4f first_traj = Matrix4f::Identity();
	for (int i = 0; i < 3000; ++i)
	{

		//Finding proper trajectory
		double timestamp = depth_timestamps[i];
		double min = std::numeric_limits<double>::infinity();
		int idx = 0;
		for (unsigned int j = 0; j < trajectories.size(); ++j)
		{
			double d = abs(trajectory_timestamps[j] - timestamp);
			if (min > d)
			{
				min = d;
				idx = j;
			}
		}

		if( i == 0 ) first_traj = trajectories[idx];
		const auto trajectory = trajectories[idx];
		
		//dllMain(pc, img, pose);
		memcpy(pose, first_traj.data(), 16 * sizeof(float));
		dllMain(pc, img, pose);

		cv::Mat dllmat = cv::Mat(getImageHeight(pc), getImageWidth(pc), CV_8UC3, img);
		imshow("dllTest", dllmat);
		cv::waitKey(1);

		Matrix4f matPose = Map<Matrix4f>(pose, 4, 4);

		//Compute the inverse of the pose
		//matPose = matPose.inverse().eval();

		std::cout << "\n ------- pose: " << i << " -------- \n" << matPose
			<< "\n------------------------ " << std::endl;

		std::cout << "\n ------- trajectory: " << i << " -------- \n" << trajectory
			<< "\n------------------------ " << std::endl;

		//Error using Frobenius norm
		//Performance metric should be Absolute Trajectory Error (ATE) https://vision.in.tum.de/data/datasets/rgbd-dataset/tools#evaluation

		Matrix4f error = matPose - trajectory;

		std::cout << "\n ------- Error: " << i << " -------- \n" << error.norm()
			<< "\n------------------------ " << std::endl;

		std::cin.get();
	}
}
