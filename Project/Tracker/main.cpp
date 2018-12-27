#include <iostream>

#include "Context.h"
#include "Tracker.h"
#include "DatasetVideoStreamReader.h"
#include "ExportDLL.h"

void vidReadTest();

void dllVidReadTest();

int main(int argc, char **argv) {

    dllVidReadTest();

    Tracker tracker;

    tracker.computerCameraPose(nullptr, nullptr, 0, 0);

    std::cin.get();

    return 0;
}

//-------------------------------TESTS---------------------------------


void vidReadTest() {

    VideoStreamReaderBase *videoInputReader = new DatasetVideoStreamReader(DATASET_DIR, false);

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

#ifdef _WIN32
void dllVidReadTest() {

	Context *pc = (Context *)createContext();

	byte *img = new byte[getImageWidth(pc) * getImageHeight(pc) * 3];

	float pose[16];

	dllMain(pc, img, pose);

	cv::Mat dllmat = cv::Mat(getImageHeight(pc), getImageWidth(pc), CV_8UC3, img);


	cv::imshow("dllTest", dllmat);
	cv::waitKey(1);
}

#else
void dllVidReadTest() {

	std::cout << "This is a Windows only test" << std:endl; 
}
#endif // _WIN32


