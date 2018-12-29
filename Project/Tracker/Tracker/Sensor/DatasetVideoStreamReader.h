#pragma once

#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <utility>

#include "VideoStreamReaderBase.h"

class DatasetVideoStreamReader :
        public VideoStreamReaderBase {
public:
    explicit DatasetVideoStreamReader(std::string datasetFolderPath, bool realtime = false) :
            m_datasetFolderPath(std::move(datasetFolderPath)),
            m_realtime(realtime) {
    }

    ~DatasetVideoStreamReader() override = default;

    // Inherited via VideoStreamReader
    bool startReading() override;

    bool stopReading() override;


protected:

    bool nextFrameAvailable() override;

    int getSequentialFrame(cv::Mat &rgb, cv::Mat &depth) override;

    int getLatestFrame(cv::Mat &rgb, cv::Mat &depth) override;

private:

    int readAnyFrame(const unsigned long &index, cv::Mat &rgb, cv::Mat &depth);

    std::string m_datasetFolderPath;
    bool m_realtime;
    unsigned long m_numFrames = 0;

    std::chrono::high_resolution_clock::time_point m_startTime;
    std::vector<std::pair<double, std::string>> m_rgb_names;
    std::vector<std::pair<double, std::string>> m_depth_names;
};