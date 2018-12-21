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

	std::chrono::duration<double> timeDiff = m_startTime - std::chrono::high_resolution_clock::now(); 
	double timeDiffSec = std::chrono::duration_cast<std::chrono::seconds>(timeDiff).count(); 

	if (m_realtime && timeDiffSec < m_rgb_names[getCurrentFrameIndex() +1].first - m_rgb_names[0].first)
		return false; 

	return true;
}

int DatasetVideoStreamReader::getSequentialFrame(unsigned char ** rgb, float ** depth)
{
	
	return readAnyFrame(getCurrentFrameIndex() + 1, rgb, depth); 
}

int DatasetVideoStreamReader::getLatestFrame(unsigned char ** rgb, float ** depth)
{
	std::chrono::duration<double> timeDiff = m_startTime - std::chrono::high_resolution_clock::now();
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

int DatasetVideoStreamReader::readAnyFrame(const unsigned long & index, unsigned char ** rgb, float ** depth)
{
	assert(index <= m_numFrames); 

	throw("Not implemneted"); 

	return 0;
}
