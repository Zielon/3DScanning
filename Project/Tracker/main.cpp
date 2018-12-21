
#include <iostream>

#include "Tracker.h"

#include "DatasetVideoStreamReader.h" 


const std::string DATASET_DIR = "C:/Users/radne/Desktop/3ds/data/rgbd_dataset_freiburg1_rpy/"; 


int main(int argc, char** argv) 
{

	DatasetVideoStreamReader *videoInputReader = new DatasetVideoStreamReader(DATASET_DIR, 30);
    Tracker tracker;

    tracker.computerCameraPose(nullptr, nullptr, 0, 0);


	std::cin.get(); 

    return 0;
}