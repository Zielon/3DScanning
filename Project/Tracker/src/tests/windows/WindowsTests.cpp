#include "WindowsTests.h"
#include "../../TrackerContext.h"

#ifdef _WIN32

void WindowsTests::run(){
    dllVidReadTest();
    vidReadTest();
}

void WindowsTests::dllVidReadTest() {

    std::cout << "START dllVidReadTest()" << std::endl;

	TrackerContext *pc = static_cast<TrackerContext*>(createContext());

    byte *img = new byte[getImageWidth(pc) * getImageHeight(pc) * 3];

    float pose[16];


	for (int i = 0; i < 120; ++i)
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

#endif