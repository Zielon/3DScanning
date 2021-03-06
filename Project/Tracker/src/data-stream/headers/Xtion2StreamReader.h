/*
 * Course: 3D Scanning and Motion Capture.
 * File: Xtion2StreamReader.h
 * Purpose: First exercise of the course.
 * @author Baris Yazici, Juan Raul Padron Griffe, Patrick Radner, Wojciech Zielonka.
 * @version 1.0 4/1/2019
Sensor specifications: https://www.asus.com/3D-Sensor/Xtion_PRO/specifications/
*/

#ifndef XTION2_STREAM_READER_H

#define XTION2_STREAM_READER_H

#define READ_WAIT_TIMEOUT 2000

#include "VideoStreamReaderBase.h"
#include <iostream>

//------------------OpenNI 2------------------

#include <OpenNI.h>

class Xtion2StreamReader : public VideoStreamReaderBase {

public:
	Xtion2StreamReader(bool realtime = true, bool verbose = false, bool capture = false);

	~Xtion2StreamReader() override;

	// Inherited via VideoStreamReader
	bool startReading() override;

	bool stopReading() override;

	bool isRunning() override;

	Matrix3f getCameraIntrinsics() override;

protected:

	bool nextFrameAvailable() override;

	int getSequentialFrame(cv::Mat &rgb, cv::Mat &depth) override;

	int getLatestFrame(cv::Mat &rgb, cv::Mat &depth) override;

private:

	const std::string m_DATA_DIR = "data";
	bool m_realtime;
	bool m_use_capture;
	bool m_use_verbose;
	openni::Device m_device;
	openni::VideoStream m_color_stream;
	openni::VideoStream m_depth_stream;

	//Camera specifications
	float m_fov_x;//Horizontal field of view
	float m_fov_y;//Horizontal field of view

	//Methods
	int readFrame(cv::Mat &rgb, cv::Mat &depth);
	bool saveFrame(int frame, cv::Mat &rgb, cv::Mat &depth);
	float computeFocalLengthRadians(float fov, float center);

	bool initContext();
};

#endif XTION2_STREAM_READER_H
