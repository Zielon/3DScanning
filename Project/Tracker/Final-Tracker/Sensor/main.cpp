#include "../../src/data-stream/headers/XtionStreamReader.h";
#include "../../src/data-stream/headers/Xtion2StreamReader.h";

#include <conio.h>
#include <opencv2/imgproc/imgproc.hpp>

int wasKeyboardHit()
{
	return (int)_kbhit();
}

int main() {

	//Sensor Class using OpenNI 2
	#if _DEBUG
		Xtion2StreamReader *streamReader = new Xtion2StreamReader(true, true, true);
	#else
		Xtion2StreamReader *streamReader = new Xtion2StreamReader(true, false, false);
	#endif

	if (!streamReader->initContext()) {
		std::cout << "Failed to create input stream context" << std::endl;
		cin.get();
		return -1;
	}

	std::cout << "Stream created properly" << std::endl;
	//cin.get();

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

	//Sensor Class using OpenNI 1

	/*while (!xnOSWasKeyboardHit())
	{

		std::cout << "Frame: " << ++i << std::endl;

		cv::Mat rgb;
		cv::Mat depth;
		streamReader->getNextFrame(rgb, depth, false);

		//Debug color image
		cv::imshow("TestRGB", rgb);

		//Debug depth image
		double min;
		double max;
		cv::minMaxIdx(depth, &min, &max);
		cv::Mat adjMap;
		cv::convertScaleAbs(depth, adjMap, 255 / max);
		cv::imshow("TestDepth", adjMap);

		cv::waitKey(1);
	}*/

	delete streamReader;

	return 0;
}