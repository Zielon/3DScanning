#include <XtionStreamReader.h>

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

	m_realtime = realtime;
	use_capture = true;
	use_verbose = false;
}

XtionStreamReader::~XtionStreamReader(){
	
	//Release resources
	color_generator.Release();
	depth_generator.Release();
	scriptNode.Release();
	context.Release();
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

	return readFrame(rgb, depth);
}

int XtionStreamReader::getLatestFrame(cv::Mat &rgb, cv::Mat &depth) {

	return readFrame(rgb, depth);
}

bool XtionStreamReader::startReading() {

	XnStatus nRetVal = XN_STATUS_OK;

	// Setting image generator(RGB color)
	nRetVal = context.FindExistingNode(XN_NODE_TYPE_IMAGE, color_generator);
	CHECK_RC(nRetVal, "Find color generator");

	//Setting depth degenerator
	nRetVal = context.FindExistingNode(XN_NODE_TYPE_DEPTH, depth_generator);
	CHECK_RC(nRetVal, "Find depth generator");
	
	xn::ImageMetaData colorMD;
	xn::DepthMetaData depthMD;
	
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

	//FPS initialization
	nRetVal = xnFPSInit(&xnFPS, 180);
	CHECK_RC(nRetVal, "FPS Init");

	return true;
}

bool XtionStreamReader::stopReading() {

	return true;
}


int XtionStreamReader::readFrame(cv::Mat &rgb, cv::Mat &depth) {

	//Read a new frame
	XnStatus nRetVal = context.WaitAnyUpdateAll();

	if (nRetVal != XN_STATUS_OK)
	{
		printf("ReadData failed: %s\n", xnGetStatusString(nRetVal));
		return -1;
	}

	xnFPSMarkFrame(&xnFPS);

	//Getting data from generator
	xn::ImageMetaData colorMD;
	xn::DepthMetaData depthMD;

	color_generator.GetMetaData(colorMD);
	depth_generator.GetMetaData(depthMD);

	const unsigned char *color_map = colorMD.Data();
	const unsigned short *depth_map = depthMD.Data();

	if (use_verbose) {
		printf("Color frame %d: resolution (%d, %d), bytes %d\n", colorMD.FrameID(), colorMD.XRes(), colorMD.YRes(), colorMD.DataSize());
		printf("Depth frame %d: resolution (%d, %d), bytes %d\n", depthMD.FrameID(), depthMD.XRes(), depthMD.YRes(), depthMD.DataSize());
	}

	//OpenCV color image from raw color map

	rgb = cv::Mat(colorMD.YRes(), colorMD.XRes(), CV_8UC3, (void*)color_map, cv::Mat::AUTO_STEP);
	/*rgb = cv::Mat(colorMD.YRes(), colorMD.XRes(), CV_8UC3);
	memcpy(rgb.data, color_map, colorMD.YRes() * colorMD.XRes() * 3 * sizeof(unsigned char));*/

	//OpenCV depth image from raw depth map
	depth = cv::Mat(depthMD.YRes(), depthMD.XRes(), CV_16UC1, (void*)depth_map, cv::Mat::AUTO_STEP);

	/*depth = cv::Mat(depthMD.YRes(), depthMD.XRes(), CV_16UC1);
	memcpy(depth.data, depth_map, depthMD.YRes() * depthMD.XRes() * sizeof(unsigned short));*/

	//Capture frames

	if (use_capture) {

		//saveRawFrame(colorMD.FrameID(), &colorMD, &depthMD);
		saveFrame(colorMD.FrameID(), rgb, depth);
	}

	depth.convertTo(depth, CV_8U, 255);
}

bool XtionStreamReader::saveRawFrame(int frame, xn::ImageMetaData *colorMD, xn::DepthMetaData *depthMD) {

	char path[100] = "";

	sprintf_s(path, "%s\\color_map_%d.raw", DATA_DIR.c_str(), frame);
	xnOSSaveFile(path, colorMD->Data(), colorMD->DataSize());

	sprintf_s(path, "%s\\depth_map_%d.raw", DATA_DIR.c_str(), frame);
	xnOSSaveFile(path, depthMD->Data(), depthMD->DataSize());

	return true;
}

bool XtionStreamReader::saveFrame(int frame, cv::Mat &rgb, cv::Mat &depth) {

	char path[100] = "";

	sprintf_s(path, "%s/color_map_%d.png", DATA_DIR.c_str(), frame);
	cv::imwrite(path, rgb);

	sprintf_s(path, "%s/depth_map_%d.png", DATA_DIR.c_str(), frame);
	cv::imwrite(path, depth);

	return true;
}

Matrix3f XtionStreamReader::getCameraIntrinsics()
{
	Matrix3f i;
	i << 520.9, 0, 325.1,
		0, 521.0, 249.7,
		0, 0, 0;
	return i;
}