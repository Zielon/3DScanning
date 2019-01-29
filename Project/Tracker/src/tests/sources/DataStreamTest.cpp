#include "../headers/DataStreamTest.h"

void DataStreamTest::vidReadTest(){
	std::cout << "START vidReadTest()" << std::endl;

	VideoStreamReaderBase* videoInputReader = new DatasetVideoStreamReader(m_params->m_dataset_path, false);

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

int DataStreamTest::sensorTest(bool useOpenni2)
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
		//streamReader = new XtionStreamReader(path.c_str(), realtime, verbose, capture);
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
