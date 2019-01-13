#include "WindowsTests.h"
#include "../../TrackerContext.h"

#include <fstream>
#include <sstream>
#include <direct.h>
#include <io.h>
#include "../../debugger/headers/Verbose.h"

// The path to the DATASET dir, must end with a backslash!
const std::string DATASET_DIR = "\\..\\..\\..\\MarkerlessAR_Unity\\Datasets\\freiburg\\";

void WindowsTests::run(){
	 reconstructionTest();
	// meshTest();
	// vidReadTest();
	// cameraPoseTest();
}

void WindowsTests::meshTest(){

	std::cout << "START meshTest()" << std::endl;

	char cCurrentPath[FILENAME_MAX];

	_getcwd(cCurrentPath, sizeof(cCurrentPath));

	strcpy(cCurrentPath + strlen(cCurrentPath), DATASET_DIR.c_str());

	TrackerContext* pc = static_cast<TrackerContext*>(createContext(cCurrentPath));

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

void WindowsTests::reconstructionTest(){

	std::cout << "START reconstructionTest()" << std::endl;

	char cCurrentPath[FILENAME_MAX];

	_getcwd(cCurrentPath, sizeof(cCurrentPath));

	strcpy(cCurrentPath + strlen(cCurrentPath), DATASET_DIR.c_str());

	TrackerContext* pc = static_cast<TrackerContext*>(createContext(cCurrentPath));

	unsigned char* img = new unsigned char[getImageWidth(pc) * getImageHeight(pc) * 3];

	float pose[16];

	for (int i = 0; i < 10; ++i)
	{
		Verbose::start();
		dllMain(pc, img, pose);
		Verbose::stop("Frame reconstruction " + std::to_string(i + 1));
	}

	pc->m_fusion->save("mesh");

	delete[]img;
	SAFE_DELETE(pc);
}

void WindowsTests::vidReadTest(){

	std::cout << "START vidReadTest()" << std::endl;

	char* path = new char[DATASET_DIR.length() + 1];
	strcpy(path, DATASET_DIR.c_str());

	VideoStreamReaderBase* videoInputReader = new DatasetVideoStreamReader(path, false);

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

	char cCurrentPath[FILENAME_MAX];

	_getcwd(cCurrentPath, sizeof(cCurrentPath));

	strcpy(cCurrentPath + strlen(cCurrentPath), DATASET_DIR.c_str());

	TrackerContext* pc = static_cast<TrackerContext*>(createContext(cCurrentPath));

	unsigned char* img = new unsigned char[getImageWidth(pc) * getImageHeight(pc) * 3];

	float pose[16];

	//Read groundtruth trajectories (camera poses)
	std::vector<Matrix4f> trajectories;
	std::vector<double> trajectory_timestamps;

	if (!readTrajectoryFile(string(cCurrentPath) + "groundtruth.txt", trajectories, trajectory_timestamps))
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

bool WindowsTests::readTrajectoryFile(const std::string& filename, std::vector<Matrix4f>& result,
                                      std::vector<double>& timestamps){

	std::ifstream file(filename, std::ios::in);

	if (!file.is_open()) return false;
	result.clear();

	//Skip not important lines
	std::string dump;
	std::getline(file, dump);
	std::getline(file, dump);
	std::getline(file, dump);

	while (file.good())
	{
		//Read data from file
		double timestamp;
		file >> timestamp;
		Vector3f translation;
		file >> translation.x() >> translation.y() >> translation.z();
		Quaternionf rot;
		file >> rot;

		//Build pose matrix from data
		Matrix4f transf;
		transf.setIdentity();
		transf.block<3, 3>(0, 0) = rot.toRotationMatrix();
		transf.block<3, 1>(0, 3) = translation;

		if (rot.norm() == 0) break;

		//Compute the inverse of the pose
		transf = transf.inverse().eval();

		//Save results
		timestamps.push_back(timestamp);
		result.push_back(transf);
	}

	file.close();

	return true;
}

void readTrajectories(){

	std::vector<Matrix4f> m_trajectory;
	std::vector<double> m_trajectoryTimeStamps;
}
