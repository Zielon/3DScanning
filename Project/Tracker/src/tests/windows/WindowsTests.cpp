#include "WindowsTests.h"
#include "../../TrackerContext.h"

#include <fstream>
#include <sstream>
#include <direct.h>
#include <io.h>
#include "../../debugger/headers/Verbose.h"
#include "../../reconstruction/headers/Mesh.h"

void WindowsTests::run(){
	// reconstructionTest();
	streamPointCloudTest();
	// meshTest();
	// vidReadTest();
	// cameraPoseTest();
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

	for (int i = 0; i < 10; ++i)
	{
		Verbose::start();

		double timestamp = depth_timestamps[i];
		double min = INFINITY;
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
		context->m_videoStreamReader->getNextFrame(rgb, depth, false);
		PointCloud* source = new PointCloud(context->m_tracker->getCameraParameters(), depth, rgb, true);
		source->transformToWorldSpace(trajectory);
		Mesh(source).save("point_cloud_" + std::to_string(i + 1));
		SAFE_DELETE(source);

		Verbose::stop("Point cloud generated " + std::to_string(i + 1), WARNING);
	}

	delete[]img;
	SAFE_DELETE(context);
}

void WindowsTests::reconstructionTest(){

	Verbose::message("START reconstructionTest()");

	TrackerContext* context = static_cast<TrackerContext*>(createContext(DatasetManager::getCurrentPath().data()));

	unsigned char* img = new unsigned char[getImageWidth(context) * getImageHeight(context) * 3];

	float pose[16];

	for (int i = 0; i < 10; ++i)
	{
		Verbose::start();
		dllMain(context, img, pose);
		Verbose::stop("Frame reconstruction " + std::to_string(i + 1), SUCCESS);
	}

	context->m_fusion->save("mesh");

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

bool WindowsTests::cameraPoseTest(){

	std::cout << "START cameraPoseTest()" << std::endl;

	TrackerContext* pc = static_cast<TrackerContext*>(createContext(DatasetManager::getCurrentPath().data()));

	unsigned char* img = new unsigned char[getImageWidth(pc) * getImageHeight(pc) * 3];

	float pose[16];

	//Read groundtruth trajectories (camera poses)
	std::vector<Matrix4f> trajectories;
	std::vector<double> trajectory_timestamps;

	if (!m_files_manager.readTrajectoryFile(trajectories, trajectory_timestamps))
	{
		std::cout << "Groundtruth trajectories are not available" << std::endl;
		return false;
	}

	for (int i = 0; i < 3000; ++i)
	{
		dllMain(pc, img, pose);

		cv::Mat dllmat = cv::Mat(getImageHeight(pc), getImageWidth(pc), CV_8UC3, img);
		imshow("dllTest", dllmat);
		cv::waitKey(1);

		Matrix4f matPose = Map<Matrix4f>(pose, 4, 4);

		//Compute the inverse of the pose
		//matPose = matPose.inverse().eval();

		std::cout << "\n ------- pose: " << i << " -------- \n" << matPose
			<< "\n------------------------ " << std::endl;

		std::cout << "\n ------- trajectory: " << i << " -------- \n" << trajectories[i]
			<< "\n------------------------ " << std::endl;

		//Error using Frobenius norm
		//Performance metric should be Absolute Trajectory Error (ATE) https://vision.in.tum.de/data/datasets/rgbd-dataset/tools#evaluation

		Matrix4f error = matPose - trajectories[i];

		std::cout << "\n ------- Error: " << i << " -------- \n" << error.norm()
			<< "\n------------------------ " << std::endl;

		std::cin.get();
	}
}
