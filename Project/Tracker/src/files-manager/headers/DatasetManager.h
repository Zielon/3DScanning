#ifndef TRACKER_LIB_FILE_MANAGER_H
#define TRACKER_LIB_FILE_MANAGER_H

#include <fstream>
#include <sstream>
#include <direct.h>
#include <vector>
#include "../../Eigen.h"

class DatasetManager
{
public:

	DatasetManager(std::string _DATASET_DIR = "\\..\\..\\..\\MarkerlessAR_Unity\\Datasets\\freiburg\\"){
		DATASET_DIR = _DATASET_DIR;
	}

	bool readTrajectoryFile(std::vector<Matrix4f>& result, std::vector<double>& timestamps);
	
	bool readDepthTimeStampFile(std::vector<double>& timestamps);

	std::string getCurrentPath();
	std::string getCurrentPath(std::string filename);

private:
	std::string DATASET_DIR; 
};

#endif
