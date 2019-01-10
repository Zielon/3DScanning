#include "WindowsTests.h"
#include "../../TrackerContext.h"

#include <fstream>
#include <sstream>
#include <direct.h>

// path to the dataset dir, must end with a backslash

const std::string DATASET_DIR = "\\..\\..\\..\\MarkerlessAR_Unity\\Datasets\\freiburg\\";

void WindowsTests::run(){
//	meshTest(); 
//	  dllVidReadTest();
//   vidReadTest();
	sensorTest();
}

void WindowsTests::meshTest() {

	std::cout << "START dllVidReadTest()" << std::endl;

	char cCurrentPath[FILENAME_MAX];

	_getcwd(cCurrentPath, sizeof(cCurrentPath));

	strcpy(cCurrentPath + strlen(cCurrentPath), DATASET_DIR.c_str());

	TrackerContext *pc = static_cast<TrackerContext*>(createContext(cCurrentPath));

	unsigned char *img = new unsigned char[getImageWidth(pc) * getImageHeight(pc) * 3];

	float pose[16];

	for (int i = 0; i < 3000; ++i)
	{
		dllMain(pc, img, pose);

		cv::Mat dllmat = cv::Mat(getImageHeight(pc), getImageWidth(pc), CV_8UC3, img);
		cv::imshow("dllTest", dllmat);
		cv::waitKey(1);

		Eigen::Matrix4f matPose = Map<Matrix4f>(pose, 4, 4);

		std::string filename = "meshTest"; 
		filename +=std::to_string( pc->videoStreamReader->getCurrentFrameIndex()); 
		filename += ".off"; 
		std::ofstream outFile(filename);
		if (!outFile.is_open()) continue;



		// write header
		outFile << "COFF" << std::endl;
		outFile << "# numVertices numFaces numEdges" << std::endl;
		outFile << getVertexCount(pc) << " ";
		assert(getIndexCount(pc) % 3 == 0); 
		outFile << getIndexCount(pc)/3 << " 0" << std::endl;


		outFile << "# list of vertices\n# X Y Z R G B A" << std::endl;

		float * vertexBuffer = new float[3 * getVertexCount(pc)];
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
		int * indexbuffer = new int[ getIndexCount(pc)];
		getIndexBuffer(pc, indexbuffer);
		for (size_t i = 0; i < getIndexCount(pc)/3; ++i)
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
}


void WindowsTests::dllVidReadTest() {

    std::cout << "START dllVidReadTest()" << std::endl;

	char cCurrentPath[FILENAME_MAX];

	_getcwd(cCurrentPath, sizeof(cCurrentPath));

	strcpy(cCurrentPath + strlen(cCurrentPath), DATASET_DIR.c_str()); 

	TrackerContext *pc = static_cast<TrackerContext*>(createContext(cCurrentPath));

	unsigned char *img = new unsigned char[getImageWidth(pc) * getImageHeight(pc) * 3];

    float pose[16];

	for (int i = 0; i < 3000; ++i)
	{
		dllMain(pc, img, pose);

		cv::Mat dllmat = cv::Mat(getImageHeight(pc), getImageWidth(pc), CV_8UC3, img);
		cv::imshow("dllTest", dllmat);
		cv::waitKey(1);

		Eigen::Matrix4f matPose = Map<Matrix4f>(pose, 4, 4);

		std::cout << "\n ------- pose: " << i << " -------- \n" << matPose
			<< "\n------------------------ "<< std::endl; 
	}
}

void WindowsTests::vidReadTest() {

    std::cout << "START vidReadTest()" << std::endl;

	char * path = new char[DATASET_DIR.length() + 1];
	strcpy(path, DATASET_DIR.c_str());

    VideoStreamReaderBase *videoInputReader = new DatasetVideoStreamReader(path, false);

    if (!videoInputReader->startReading()) {
        std::cout << "Failed to read input stream" << std::endl;
        exit(-1);
    }

    cv::Mat rgb;
    cv::Mat depth;

    for (int i = 0; i < 3000; ++i) {

        videoInputReader->getNextFrame(rgb, depth, true);

        cv::imshow("TestRGB", rgb);
        double min;
        double max;
        cv::minMaxIdx(depth, &min, &max);
        cv::Mat adjMap;
        cv::convertScaleAbs(depth, adjMap, 255 / max);
        cv::imshow("TestDepth", adjMap);

        cv::waitKey(1);

    }
}


int WindowsTests::sensorTest() {

	string path = "";

	XtionStreamReader *streamReader = new XtionStreamReader(path.c_str(), true, false, false);

	if (!streamReader->initContext()) {
		std::cout << "Failed to create input stream context" << std::endl;
		return -1;
	}

	std::cout << "Stream created properly" << std::endl;

	if (!streamReader->startReading()) {
		std::cout << "Failed to read input stream" << std::endl;
		return -1;
	}

	Matrix3f intrinsics = streamReader->getCameraIntrinsics();

	std::cout << "Sensor intrinsics: " << std::endl << intrinsics << std::endl;

	std::cout << "The reading process has started" << std::endl;

	//cin.get();

	int i = 0;

	while (!xnOSWasKeyboardHit())
	{

		std::cout << "Frame: " << ++i << std::endl;

		cv::Mat rgb;
		cv::Mat depth;
		streamReader->getNextFrame(rgb, depth, false);

		cv::cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);

		//Debug color image
		cv::imshow("TestRGB", rgb);

		//Debug depth image
		double min;
		double max;
		cv::minMaxIdx(depth, &min, &max);
		cv::Mat adjMap;
		cv::convertScaleAbs(depth, adjMap, 255 / max);
		cv::imshow("TestDepth", adjMap);


		/* Raw depth image
		depth.convertTo(depth, CV_8U, 255);
		cv::imshow("TestDepth", depth);*/

		cv::waitKey(1);
	}

	delete streamReader;
}