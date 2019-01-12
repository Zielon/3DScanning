/*
 * Course: 3D Scanning and Motion Capture.
 * File: XtionStreamReader.h
 * Purpose: First exercise of the course.
 * @author Baris Yazici, Juan Raul Padron Griffe, Patrick Radner, Wojciech Zielonka.
 * @version 1.0 4/1/2019
Sensor specifications: https://www.asus.com/3D-Sensor/Xtion_PRO/specifications/
*/


#ifndef XTION_STREAM_READER_H

#define XTION_STREAM_READER_H

#include "VideoStreamReaderBase.h"
#include <iostream>
#include <fstream>

//OpenNI

//---------------------------------------------------------------------------
// Macros
//---------------------------------------------------------------------------
#define CHECK_RC(rc, what)											\
	if (rc != XN_STATUS_OK)											\
	{																\
		printf("%s failed: %s\n", what, xnGetStatusString(rc));		\
		return rc;													\
	}

#define OPENNI_XML_PATH "OpenNIConfig.xml"

#include <XnOpenNI.h>
#include <XnLog.h>
#include <XnCppWrapper.h>
#include <XnFPSCalculator.h>

using namespace std;
using namespace xn;

class XtionStreamReader : public VideoStreamReaderBase {

public:
	XtionStreamReader(bool realtime = true, bool verbose = false, bool capture = false);

	~XtionStreamReader() override;

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
	xn::Context m_context;
	xn::ScriptNode m_scriptNode;
	xn::ImageGenerator m_color_generator;
	xn::DepthGenerator m_depth_generator;
	XnFPSData xnFPS;

	//Camera specifications
	const float m_fov_x_degrees = 58.0f;//Horizontal field of view in degrees (58 degrees)
	const float m_fov_y_degrees = 45.0f;//Vertical field of view in degrees (47 degrees)
	float m_fov_x;//Horizontal field of view
	float m_fov_y;//Horizontal field of view
	float m_x_res;//Horizontal resolution
	float m_y_res;//Vertical resolution

	//Methods
	XnBool fileExists(const char *fn);
	int readFrame(cv::Mat &rgb, cv::Mat &depth);
	bool saveRawFrame(int frame, xn::ImageMetaData *colorMD, xn::DepthMetaData *depthMD);
	bool saveFrame(int frame, cv::Mat &rgb, cv::Mat &depth);
	float computeFocalLength(float fov_angle, float center);
	float computeFocalLengthRadians(float fov, float center);
	bool initContext();
};

#endif XTION_STREAM_READER_H