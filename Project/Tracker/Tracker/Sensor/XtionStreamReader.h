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

#define SAMPLE_XML_PATH "SamplesConfig.xml"

#include <XnOpenNI.h>
#include <XnLog.h>
#include <XnCppWrapper.h>
#include <XnFPSCalculator.h>

using namespace std;
using namespace xn;

class XtionStreamReader : public VideoStreamReaderBase {

public:
	XtionStreamReader(bool realtime = false);

	~XtionStreamReader() override;

	// Inherited via VideoStreamReader
	bool startReading() override;

	bool stopReading() override;

protected:

	bool nextFrameAvailable() override;

	int getSequentialFrame(cv::Mat &rgb, cv::Mat &depth) override;

	int getLatestFrame(cv::Mat &rgb, cv::Mat &depth) override;

private:

	bool m_realtime;
	xn::Context context;
	xn::ScriptNode scriptNode;
	xn::ImageGenerator color_generator;
	xn::ImageMetaData colorMD;
	xn::DepthGenerator depth_generator;
	xn::DepthMetaData depthMD;
	XnFPSData xnFPS;

	XnBool fileExists(const char *fn);
	int readFrame(cv::Mat &rgb, cv::Mat &depth);

};

#endif XTION_STREAM_READER_H
