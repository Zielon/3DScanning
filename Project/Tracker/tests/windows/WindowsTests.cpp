#include "WindowsTests.h"

#ifdef _WIN32

void WindowsTests::run(){
    dllVidReadTest();
    vidReadTest();
}

void WindowsTests::dllVidReadTest() {

    std::cout << "START dllVidReadTest()" << std::endl;

    Context *pc = (Context *)createContext();

    byte *img = new byte[getImageWidth(pc) * getImageHeight(pc) * 3];

    float pose[16];

    dllMain(pc, img, pose);

    cv::Mat dllmat = cv::Mat(getImageHeight(pc), getImageWidth(pc), CV_8UC3, img);
    cv::imshow("dllTest", dllmat);
    cv::waitKey(1);
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