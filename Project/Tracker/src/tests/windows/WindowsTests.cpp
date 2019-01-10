#include "WindowsTests.h"
#include "../../TrackerContext.h"

#include <fstream>
#include <sstream>
#include <direct.h>
#include <io.h>
#include "../../reconstruction/headers/Mesh.h"
#include "../../reconstruction/headers/MarchingCubes.h"

// The path to the DATASET dir, must end with a backslash!
const std::string DATASET_DIR = "\\..\\..\\..\\MarkerlessAR_Unity\\Datasets\\freiburg\\";

void WindowsTests::run(){
	// meshTest();
	dllVidReadTest();
	// vidReadTest();
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

void WindowsTests::dllVidReadTest(){

	std::cout << "START dllVidReadTest()" << std::endl;

	char cCurrentPath[FILENAME_MAX];

	_getcwd(cCurrentPath, sizeof(cCurrentPath));

	strcpy(cCurrentPath + strlen(cCurrentPath), DATASET_DIR.c_str());

	TrackerContext* pc = static_cast<TrackerContext*>(createContext(cCurrentPath));

	unsigned char* img = new unsigned char[getImageWidth(pc) * getImageHeight(pc) * 3];

	float pose[16];

	for (int i = 0; i < 3; ++i)
	{
		dllMain(pc, img, pose);

		cv::Mat dllmat = cv::Mat(getImageHeight(pc), getImageWidth(pc), CV_8UC3, img);
		imshow("dllTest", dllmat);
		cv::waitKey(1);

		Matrix4f matPose = Map<Matrix4f>(pose, 4, 4);

		std::cout << "\n ------- pose: " << i << " -------- \n" << matPose
			<< "\n------------------------ " << std::endl;
	}

	Mesh mesh;

	#pragma omp parallel
	for (unsigned int x = 0; x < pc->m_fusion->m_volume_size - 1; x++)
		for (unsigned int y = 0; y < pc->m_fusion->m_volume_size - 1; y++)
			for (unsigned int z = 0; z < pc->m_fusion->m_volume_size - 1; z++)
				ProcessVolumeCell(pc->m_fusion->getVolume(), x, y, z, 0.00f, &mesh);
			

	mesh.WriteMesh("mesh.off");

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

	SAFE_DELETE(videoInputReader);
}
