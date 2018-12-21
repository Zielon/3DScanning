
#include <iostream>

#include "Tracker.h"

#include "DatasetVideoStreamReader.h" 


const std::string DATASET_DIR = "D:/user/desktop/3dscanning/assets/rgbd_dataset_freiburg2_xyz/"; 


int main(int argc, char** argv) 
{

	VideoStreamReader *videoInputReader = new DatasetVideoStreamReader(DATASET_DIR, true);


	if (!videoInputReader->startReading())
	{
		std::cout << "Failed to read input stream" << std::endl; 
		exit(-1); 
	}


	cv::Mat rgb;
	cv::Mat depth;


	for (int i = 0; i < 3000; ++i)
	{

		videoInputReader->getNextFrame(rgb, depth, false);

		cv::imshow("TestRGB", rgb);
		double min;
		double max;
		cv::minMaxIdx(depth, &min, &max);
		cv::Mat adjMap;
		cv::convertScaleAbs(depth, adjMap, 255 / max);
		cv::imshow("TestDepth", adjMap);

		cv::waitKey(1); 

	}

    Tracker tracker;

    tracker.computerCameraPose(nullptr, nullptr, 0, 0);


	std::cin.get(); 

    return 0;
}