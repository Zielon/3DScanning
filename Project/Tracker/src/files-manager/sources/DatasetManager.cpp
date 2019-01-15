#include "../headers/DatasetManager.h"


bool DatasetManager::readTrajectoryFile(std::vector<Matrix4f>& result, std::vector<double>& timestamps){

	std::ifstream file(getCurrentPath("groundtruth.txt"), std::ios::in);

	if (!file.is_open()) return false;

	result.clear();

	//Skip not important lines
	std::string dump;
	std::getline(file, dump);
	std::getline(file, dump);
	std::getline(file, dump);

	while (file.good())
	{
		//Read data from file
		double timestamp;
		file >> timestamp;
		Vector3f translation;
		file >> translation.x() >> translation.y() >> translation.z();
		Quaternionf rot;
		file >> rot;

		//Build pose matrix from data
		Matrix4f trajectory;
		trajectory.setIdentity();
		trajectory.block<3, 3>(0, 0) = rot.toRotationMatrix();
		trajectory.block<3, 1>(0, 3) = translation;

		if (rot.norm() == 0) break;

		trajectory = trajectory.eval();

		//Save results
		timestamps.push_back(timestamp);
		result.push_back(trajectory);
	}

	file.close();

	return true;
}

bool DatasetManager::readDepthTimeStampFile(std::vector<double>& timestamps){

	std::ifstream fileDepthList(getCurrentPath("depth.txt"), std::ios::in);
	if (!fileDepthList.is_open()) return false;
	timestamps.clear();

	std::string dump;
	std::getline(fileDepthList, dump);
	std::getline(fileDepthList, dump);
	std::getline(fileDepthList, dump);
	while (fileDepthList.good())
	{
		double timestamp;
		fileDepthList >> timestamp;
		std::string filename;
		fileDepthList >> filename;
		if (filename == "") break;
		timestamps.push_back(timestamp);
	}
	fileDepthList.close();
	return true;
}

std::string DatasetManager::getCurrentPath(){
	char current[FILENAME_MAX];
	_getcwd(current, sizeof(current));
	strcpy(current + strlen(current), DATASET_DIR.c_str());
	return std::string(DATASET_DIR);
}

std::string DatasetManager::getCurrentPath(std::string filename){
	return getCurrentPath() + "\\" + filename;
}
