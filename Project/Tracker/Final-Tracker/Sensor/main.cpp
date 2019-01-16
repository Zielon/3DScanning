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
		VideoStreamReaderBase *streamReader = new Xtion2StreamReader(true, true, true);
		//VideoStreamReaderBase *streamReader = new XtionStreamReader(true, true, true);
	#else
		VideoStreamReaderBase *streamReader = new Xtion2StreamReader(true, false, false);
		//VideoStreamReaderBase *streamReader = new XtionStreamReader(true, false, false);
	#endif

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

		cv::Mat rgb, depth, scaledDepth, filteredDepth;

		streamReader->getNextFrame(rgb, depth, false);

		//Debug color image
		cv::cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);
		cv::imshow("RGB Color", rgb);

		//Debug depth image

		//Scale depth map to show as a image
		double min;
		double max;
		cv::minMaxIdx(depth, &min, &max);
		cv::convertScaleAbs(depth, scaledDepth, 255 / max);
		cv::imshow("Raw Depth", scaledDepth);

		//cv::imwrite("raw_depth.png", scaledDepth);

		//Bilateral Filter
		cv::bilateralFilter (scaledDepth, filteredDepth, 9, 150, 150);//(9,32)
		cv::imshow("Bilateral Filter Depth", filteredDepth);

		//cv::imwrite("bilateral_depth.png", scaledDepth);

		/*cv::Mat filteredDepthTest;
		cv::medianBlur(scaledDepth, filteredDepthTest, 9);

		cv::imwrite("median_depth.png", filteredDepthTest);*/

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