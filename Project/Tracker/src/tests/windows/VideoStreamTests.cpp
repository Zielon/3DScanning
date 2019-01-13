#include "VideoStreamTests.h"

void VideoStreamTests::run() {

	sensorTest(true);
}

int VideoStreamTests::wasKeyboardHit()
{
	return (int)_kbhit();
}

int VideoStreamTests::sensorTest(bool useOpenni2)
{
	bool realtime = true, capture = false, verbose = false;
	VideoStreamReaderBase *streamReader = NULL;

	string path = "";

	#if _DEBUG
		capture = true;
		verbose = true;
	#endif

	//Sensor Class using OpenNI 2
	if (useOpenni2) {
		streamReader = new Xtion2StreamReader(realtime, verbose, capture);
	}
	else {
		streamReader = new XtionStreamReader(path.c_str(), realtime, verbose, capture);
	}

	if (!streamReader->startReading()) {
		std::cout << "Failed to read input stream" << std::endl;
		cin.get();
		return -1;
	}

	Matrix3f intrinsics = streamReader->getCameraIntrinsics();

	std::cout << "Sensor intrinsics: " << std::endl << intrinsics << std::endl;

	std::cout << "The reading process has started" << std::endl;

	int i = 0;

	while (!wasKeyboardHit())
	{
		std::cout << "Frame: " << ++i << std::endl;

		cv::Mat rgb;
		cv::Mat depth;
		streamReader->getNextFrame(rgb, depth, false);

		//Debug color image
		cv::cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);
		cv::imshow("TestRGB", rgb);

		//Debug depth image
		double min;
		double max;
		cv::minMaxIdx(depth, &min, &max);
		cv::Mat adjMap;
		cv::convertScaleAbs(depth, adjMap, 255 / max);
		cv::imshow("TestDepth", adjMap);

		cv::waitKey(1);
	}

	delete streamReader;

	return 0;
}
