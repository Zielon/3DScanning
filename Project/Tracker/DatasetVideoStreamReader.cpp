#include "DatasetVideoStreamReader.h"

#include <sstream>
#include <fstream>
#include <cassert>


DatasetVideoStreamReader::~DatasetVideoStreamReader()
{
}

bool DatasetVideoStreamReader::startReading()
{
	std::string rgbLine, depthLine; 
	std::ifstream rgbNameFile(m_datasetFolderPath + "rgb.txt"); 
	std::ifstream depthNameFile(m_datasetFolderPath + "depth.txt");

	unsigned long counter = 0; 

	if(!(rgbNameFile.is_open() && depthNameFile.is_open()))
		return false; 

	while (std::getline(rgbNameFile, rgbLine) && std::getline(depthNameFile, depthLine))
	{
		if (rgbLine.at(0) == '#') // FIXME: assuming both files have same comment header -> skip more lines in 1 file
			continue; 
		{
			std::stringstream lineSS(rgbLine);

			std::string timestampStr;
			std::getline(lineSS, timestampStr, ' ');
			std::string fileName;
			std::getline(lineSS, fileName, ' ');
			m_rgb_names.push_back(std::pair<double, std::string>(std::stod(timestampStr), fileName));
		}

		{
			std::stringstream lineSS(depthLine);

			std::string timestampStr;
			std::getline(lineSS, timestampStr, ' ');
			std::string fileName;
			std::getline(lineSS, fileName, ' ');
			m_depth_names.push_back(std::pair<double, std::string>(std::stod(timestampStr), fileName));
		}

		++counter; 
	}


	assert(counter > 0); 
	m_numFrames = counter; 

	m_startTime = std::chrono::high_resolution_clock::now();

	return true;
}

bool DatasetVideoStreamReader::stopReading()
{
	m_rgb_names.clear(); 
	m_depth_names.clear(); 
	return true;
}

bool DatasetVideoStreamReader::nextFrameAvailable()
{

	if (getCurrentFrameIndex() == m_numFrames) return false; 

	std::chrono::duration<double> timeDiff = std::chrono::high_resolution_clock::now() - m_startTime;
	double timeDiffSec = std::chrono::duration_cast<std::chrono::seconds>(timeDiff).count(); 

	if (m_realtime && timeDiffSec < m_rgb_names[getCurrentFrameIndex() +1].first - m_rgb_names[0].first)
		return false; 

	return true;
}

int DatasetVideoStreamReader::getSequentialFrame(cv::Mat& rgb, cv::Mat& depth)
{
	
	return readAnyFrame(getCurrentFrameIndex() + 1, rgb, depth); 
}

int DatasetVideoStreamReader::getLatestFrame(cv::Mat& rgb, cv::Mat& depth)
{
	std::chrono::duration<double> timeDiff = std::chrono::high_resolution_clock::now() - m_startTime;
	double timeDiffSec = std::chrono::duration_cast<std::chrono::seconds>(timeDiff).count();

	assert(timeDiffSec < m_rgb_names[getCurrentFrameIndex() + 1].first - m_rgb_names[0].first); 
	unsigned long offset = 1; 

	while (getCurrentFrameIndex() + offset < m_numFrames && 
		timeDiffSec > m_rgb_names[getCurrentFrameIndex() + offset].first - m_rgb_names[0].first)
	{
		++offset;
	}

	return readAnyFrame(getCurrentFrameIndex() + offset, rgb, depth);
}

int DatasetVideoStreamReader::readAnyFrame(const unsigned long & index, cv::Mat& rgb, cv::Mat& depth)
{
	assert(index <= m_numFrames); 

	/*/

	FreeImage rgbFrame, depthFrame; 
	rgbFrame.LoadImageFromFile(m_datasetFolderPath + m_rgb_names[index].second); 
	depthFrame.LoadImageFromFile(m_datasetFolderPath + m_depth_names[index].second);



	*rgb = new BYTE[m_width_rgb * m_height_rgb * 3]; 
	*depth = new float[m_width_depth * m_height_depth];

	for (size_t i = 0; i < m_width_rgb * m_height_rgb * 3; ++i)
	{
		BYTE c = static_cast<BYTE>(rgbFrame.data[i] * 255);
		*rgb[i] = static_cast<BYTE>(rgbFrame.data[i] * 255); 
	}
	memcpy(*depth, depthFrame.data, m_width_depth * m_height_depth);

	/**/



	rgb = cv::imread(m_datasetFolderPath + m_rgb_names[index].second); 
	cv::Mat depthTmp = cv::imread(m_datasetFolderPath + m_depth_names[index].second);
	depthTmp.convertTo(depth, CV_32FC1, 1.0/5000.0);

	//FIXME: Just assuming constant w/h 
	m_width_rgb = rgb.rows;
	m_height_rgb = rgb.cols; 
	m_width_depth = depth.rows; 
	m_height_depth = depth.cols; 
	


	//FIXME: not sure why i planned for all of these functions to return something... 
	return 0;
}
