#include "WindowsTests.h"
#include "../../TrackerContext.h"

#include <direct.h>

// path to the dataset dir, must end with a backslash

const std::string DATASET_DIR = "\\..\\..\\..\\MarkerlessAR_Unity\\Datasets\\freiburg\\";

void WindowsTests::run(){
    dllVidReadTest();
 //   vidReadTest();
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