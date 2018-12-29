#include "XtionStreamReader.h"

XtionStreamReader::XtionStreamReader(bool realtime) {

	XnStatus nRetVal = XN_STATUS_OK;

	EnumerationErrors errors;
	const char *fn = NULL;

	//Check if the configuration path exists
	if (fileExists(SAMPLE_XML_PATH)) {
		fn = SAMPLE_XML_PATH;
	}
	else {
		printf("Could not find '%s' nor '%s'. Aborting.\n", SAMPLE_XML_PATH, SAMPLE_XML_PATH);
		printf("XN Status Error: %d\n", XN_STATUS_ERROR);
	}

	printf("Reading config from: '%s'\n", fn);

	//Create context from configuration file
	nRetVal = context.InitFromXmlFile(fn, scriptNode, &errors);

	if (nRetVal == XN_STATUS_NO_NODE_PRESENT)
	{
		XnChar strError[1024];
		errors.ToString(strError, 1024);
		printf("%s\n", strError);
	}
	else if (nRetVal != XN_STATUS_OK)
	{
		printf("Open failed: %s\n", xnGetStatusString(nRetVal));
	}

	printf("XN context return value: %d\n", nRetVal);

	this->m_realtime = realtime;
}

XtionStreamReader::~XtionStreamReader(){
	
	//Release resources
	this->scriptNode.Release();
	this->context.Release();
}

XnBool XtionStreamReader::fileExists(const char *fn)
{
	XnBool exists;
	xnOSDoesFileExist(fn, &exists);
	return exists;
}

bool XtionStreamReader::nextFrameAvailable() {

	return false;
}

int XtionStreamReader::getSequentialFrame(cv::Mat &rgb, cv::Mat &depth) {

	return 0;
}

int XtionStreamReader::getLatestFrame(cv::Mat &rgb, cv::Mat &depth) {

	return 0;
}

bool XtionStreamReader::startReading() {

	XnStatus nRetVal = XN_STATUS_OK;

	// Setting image generator(RGB color)
	ImageGenerator color_generator;
	nRetVal = this->context.FindExistingNode(XN_NODE_TYPE_IMAGE, color_generator);
	//CHECK_RC(nRetVal, "Find color generator");

	//Setting depth degenerator
	nRetVal = this->context.FindExistingNode(XN_NODE_TYPE_DEPTH, depth_generator);
	//CHECK_RC(nRetVal, "Find depth generator");

	color_generator.GetMetaData(colorMD);
	depth_generator.GetMetaData(depthMD);

	//Color image must be RGBformat.
	if (colorMD.PixelFormat() != XN_PIXEL_FORMAT_RGB24)
	{
		printf("The device image format must be RGB24\n");
		return false;
	}

	// Color resolution must be equal to depth resolution
	if (colorMD.FullXRes() != depthMD.FullXRes() || colorMD.FullYRes() != depthMD.FullYRes())
	{
		printf("The device depth and image resolution must be equal!\n");
		return false;
	}

	return true;
}

bool XtionStreamReader::stopReading() {

	return true;
}