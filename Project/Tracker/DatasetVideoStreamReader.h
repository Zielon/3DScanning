#pragma once

#include <iostream>
#include <chrono>
#include <vector>

#include "VideoStreamReader.h"
#include "FreeImageHelper.h"

class DatasetVideoStreamReader :
	public VideoStreamReader
{
public:
	DatasetVideoStreamReader(std::string datasetFolderPath, bool realtime = false) :
		VideoStreamReader(),
		m_datasetFolderPath(datasetFolderPath),
		m_realtime(realtime)
	{

	}

	virtual ~DatasetVideoStreamReader();


	// Inherited via VideoStreamReader
	virtual bool startReading() override;
	virtual bool stopReading() override;


protected: 

	virtual bool nextFrameAvailable() override;
	virtual int getSequentialFrame(unsigned char ** rgb, float ** depth) override;
	virtual int getLatestFrame(unsigned char ** rgb, float ** depth) override;

private:

	int readAnyFrame(const unsigned long& index, unsigned char ** rgb, float ** depth);

	std::string m_datasetFolderPath; 
	bool m_realtime; 
	unsigned long m_numFrames = 0; 

	std::chrono::high_resolution_clock::time_point m_startTime;
	std::vector < std::pair<double, std::string>> m_rgb_names; 
	std::vector < std::pair<double, std::string>> m_depth_names;


};

