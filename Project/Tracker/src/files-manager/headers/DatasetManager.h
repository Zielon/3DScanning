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
	bool readTrajectoryFile(std::vector<Matrix4f>& result, std::vector<double>& timestamps) const;
	
	bool readDepthTimeStampFile(std::vector<double>& timestamps) const;

	static std::string getCurrentPath();

private:
	std::string getCurrentPath(std::string filename) const;
};

#endif
