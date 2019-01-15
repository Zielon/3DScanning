#include "../headers/DatasetVideoStreamReader.h"

#include <fstream>

bool DatasetVideoStreamReader::startReading() {
    std::string rgbLine, depthLine;

    std::ifstream rgbNameFile(m_datasetFolderPath + "rgb.txt");
    std::ifstream depthNameFile(m_datasetFolderPath + "depth.txt");

    unsigned long counter = 0;

    if (!rgbNameFile || !depthNameFile)
    {
		throw std::runtime_error("Could not open files!");
    }

    while (std::getline(rgbNameFile, rgbLine) && std::getline(depthNameFile, depthLine)) {
        if (rgbLine.at(0) == '#') // assuming both files have same comment header
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
    cv::Mat tmp;
    readAnyFrame(0, tmp, tmp); //FIXME: Hack to set img w/h members


    m_startTime = std::chrono::high_resolution_clock::now();

    return true;
}

bool DatasetVideoStreamReader::stopReading() {
    m_rgb_names.clear();
    m_depth_names.clear();
    return true;
}

bool DatasetVideoStreamReader::nextFrameAvailable() {

    if (getCurrentFrameIndex() == m_numFrames) return false;

    std::chrono::duration<double> timeDiff = std::chrono::high_resolution_clock::now() - m_startTime;
    double timeDiffSec = std::chrono::duration_cast<std::chrono::milliseconds>(timeDiff).count() / 1000.0;

    if (m_realtime && timeDiffSec < m_rgb_names[getCurrentFrameIndex() + 1].first - m_rgb_names[0].first)
        return false;

    return true;
}

int DatasetVideoStreamReader::getSequentialFrame(cv::Mat &rgb, cv::Mat &depth) {

    return readAnyFrame(getCurrentFrameIndex(), rgb, depth);
}

bool DatasetVideoStreamReader::isRunning() {
    return !m_rgb_names.empty();
}

Matrix3f DatasetVideoStreamReader::getCameraIntrinsics()
{
	Matrix3f i;
	
	//ROS default
	i << 525.0f, 0.0f, 319.5f,
		0.0f, 525.0f, 239.5f,
		0.0f, 0.0f, 1.0f;

	//Freiburg 1
	/*i << 517.3, 0, 318.6,
		0, 516.5, 255.3,
		0, 0, 1.0;*/

	//Freiburg 2
	/*i << 520.9, 0, 325.1,
		0, 521.0, 249.7,
		0, 0, 1.0; */


	return i; 
}

int DatasetVideoStreamReader::getLatestFrame(cv::Mat &rgb, cv::Mat &depth) {
    std::chrono::duration<double> timeDiff = std::chrono::high_resolution_clock::now() - m_startTime;
    double timeDiffSec = std::chrono::duration_cast<std::chrono::milliseconds>(timeDiff).count() / 1000.0;


    double frametime = m_rgb_names[getCurrentFrameIndex() + 1].first - m_rgb_names[0].first;

    unsigned long offset = 1;

    while (getCurrentFrameIndex() + offset < m_numFrames &&
           timeDiffSec > m_rgb_names[getCurrentFrameIndex() + offset].first - m_rgb_names[0].first) {
        ++offset;
    }

    return readAnyFrame(getCurrentFrameIndex() + offset, rgb, depth);
}

int DatasetVideoStreamReader::readAnyFrame(unsigned long index, cv::Mat &rgb, cv::Mat &depth) {
	index = std::min(index, m_numFrames - 1); //Just repeat the last frame

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


    //cv::Mat depthTmp = cv::imread(m_datasetFolderPath + m_depth_names[index].second);
	cv::Mat depthTmp = cv::imread(m_datasetFolderPath + m_depth_names[index].second, 2);//Right format 
    
	//double min, max;
	//cv::minMaxLoc(depthTmp, &min, &max);//Depth range test
	
	// depth images are scaled by 5000 (see https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
	depthTmp.convertTo(depth, CV_32FC1, 1.0 / 5000.0);//Right format is CV_16FC1

	//cv::minMaxLoc(depth, &min, &max);//Depth range test

    //ust assuming constant w/h
    m_width_rgb = rgb.cols;
    m_height_rgb = rgb.rows;
    m_width_depth = depth.cols;
    m_height_depth = depth.rows;



    //not sure why i planned for all of these functions to return something...
    return 0;
}
