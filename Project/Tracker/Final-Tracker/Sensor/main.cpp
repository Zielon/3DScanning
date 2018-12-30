#include <XtionStreamReader.h>

int main(){

	XtionStreamReader *streamReader = new XtionStreamReader(false);

	std::cout << "Stream created properly" << std::endl;

	if (!streamReader->startReading()) {
		std::cout << "Failed to read input stream" << std::endl;
		return -1;
	}

	std::cout << "The reading process has started" << std::endl;

	for (int i = 0; i < 3000; i++) {

		std::cout << "Frame: " <<  i << std::endl;

		cv::Mat rgb;
		cv::Mat depth;
		streamReader->getNextFrame(rgb, depth, false);

		//Debug color image
		cv::imshow("TestRGB", rgb);

		//Debug depth image
		/*double min;
		double max;
		cv::minMaxIdx(depth, &min, &max);
		cv::Mat adjMap;
		cv::convertScaleAbs(depth, adjMap, 255 / max);
		cv::imshow("TestDepth", adjMap);*/

		cv::imshow("TestDepth", depth);

		cv::waitKey();
	}

	delete streamReader;

	return 0;
}