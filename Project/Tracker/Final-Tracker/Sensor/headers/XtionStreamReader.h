#ifndef XTION_STREAM_READER_H

#define XTION_STREAM_READER_H

#include <iostream>
#include <fstream>
#include "VideoStreamReaderBase.h"

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

	bool initContext();

	// Inherited via VideoStreamReader
	bool startReading() override;

	bool stopReading() override;

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

	XnBool fileExists(const char *fn);
	int readFrame(cv::Mat &rgb, cv::Mat &depth);
	bool saveRawFrame(int frame, xn::ImageMetaData *colorMD, xn::DepthMetaData *depthMD);
	bool saveFrame(int frame, cv::Mat &rgb, cv::Mat &depth);


};

#endif XTION_STREAM_READER_H
