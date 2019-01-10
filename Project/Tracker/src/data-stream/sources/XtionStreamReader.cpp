#include "../Headers/XtionStreamReader.h"

XtionStreamReader::XtionStreamReader(const char *sensorFolderPath, bool realtime, bool verbose, bool capture) {

	m_sensorFolderPath = sensorFolderPath;
	m_realtime = realtime;
	m_use_capture = capture;
	m_use_verbose = verbose;

	mStatus = 0;
}


bool XtionStreamReader::initContext() {

	XnStatus nRetVal = XN_STATUS_OK;

	EnumerationErrors errors;
	const char *fn = NULL;

	std::string path = m_sensorFolderPath + string(OPENNI_XML_PATH);

	//Check if the configuration path exists
	if (fileExists( path.c_str() )) {
		fn = path.c_str();
	}
	else {
		printf("Could not find '%s'. Aborting.\n", path.c_str());
		printf("XN Status Error: %d\n", XN_STATUS_ERROR);

		m_width_rgb = -1;

		return false;
	}

	printf("Reading config from: '%s'\n", fn);

	//Create context from configuration file
	nRetVal = m_context.InitFromXmlFile(fn, m_scriptNode, &errors);

	if (nRetVal == XN_STATUS_NO_NODE_PRESENT)
	{
		XnChar strError[1024];
		errors.ToString(strError, 1024);
		printf("%s\n", strError);

		return false;
	}
	else if (nRetVal != XN_STATUS_OK)
	{
		printf("Open failed: %s\n", xnGetStatusString(nRetVal));

		return false;
	}

	printf("XN context return value: %d\n", nRetVal);

	return true;
}

XtionStreamReader::~XtionStreamReader() {

	//Release resources
	m_color_generator.Release();
	m_depth_generator.Release();
	m_scriptNode.Release();
	m_context.Release();
}

XnBool XtionStreamReader::fileExists(const char *fn)
{
	XnBool exists;
	xnOSDoesFileExist(fn, &exists);
	return exists;
}

bool XtionStreamReader::nextFrameAvailable() {

	return true;
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
	nRetVal = m_context.FindExistingNode(XN_NODE_TYPE_IMAGE, m_color_generator);
	CHECK_RC(nRetVal, "Find color generator");

	//Setting depth degenerator
	nRetVal = m_context.FindExistingNode(XN_NODE_TYPE_DEPTH, m_depth_generator);
	CHECK_RC(nRetVal, "Find depth generator");

	xn::ImageMetaData colorMD;
	xn::DepthMetaData depthMD;

	m_color_generator.GetMetaData(colorMD);
	m_depth_generator.GetMetaData(depthMD);

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

	// Setting maps parameters
	m_width_rgb = colorMD.FullXRes();
	m_height_rgb = colorMD.FullYRes();
	m_width_depth = depthMD.FullXRes();
	m_height_depth = depthMD.FullYRes();


	// Setting intrinsics parameters
	m_x_res = depthMD.FullXRes();
	m_y_res = depthMD.FullYRes();

	XnFieldOfView fov;
	m_depth_generator.GetFieldOfView(fov);//Radians

	m_fov_x = fov.fHFOV;
	m_fov_y = fov.fVFOV;

	//FPS initialization
	nRetVal = xnFPSInit(&xnFPS, 180);
	CHECK_RC(nRetVal, "FPS Init");

	return true;
}

bool XtionStreamReader::stopReading() {

	return true;
}

bool XtionStreamReader::isRunning() {
	return true;
}


int XtionStreamReader::readFrame(cv::Mat &rgb, cv::Mat &depth) {

	//Read a new frame
	XnStatus nRetVal = m_context.WaitAnyUpdateAll();

	if (nRetVal != XN_STATUS_OK)
	{
		printf("ReadData failed: %s\n", xnGetStatusString(nRetVal));
		mStatus = -1;

		return -1;
	}

	xnFPSMarkFrame(&xnFPS);

	//Getting data from generator
	xn::ImageMetaData colorMD;
	xn::DepthMetaData depthMD;

	m_color_generator.GetMetaData(colorMD);
	m_depth_generator.GetMetaData(depthMD);

	const unsigned char *color_map = colorMD.Data();
	const unsigned short *depth_map = depthMD.Data();

	if (m_use_verbose) {
		printf("Color frame %d: resolution (%d, %d), bytes %d\n", colorMD.FrameID(), colorMD.XRes(), colorMD.YRes(), colorMD.DataSize());
		printf("Depth frame %d: resolution (%d, %d), bytes %d\n", depthMD.FrameID(), depthMD.XRes(), depthMD.YRes(), depthMD.DataSize());
	}

	//OpenCV color image from raw color map

	rgb = cv::Mat(colorMD.YRes(), colorMD.XRes(), CV_8UC3, (void*)color_map, cv::Mat::AUTO_STEP);
	/*rgb = cv::Mat(colorMD.YRes(), colorMD.XRes(), CV_8UC3);
	memcpy(rgb.data, color_map, colorMD.YRes() * colorMD.XRes() * 3 * sizeof(unsigned char));*/

	//OpenCV depth image from raw depth map
	depth = cv::Mat(depthMD.YRes(), depthMD.XRes(), CV_16UC1, (void*)depth_map, cv::Mat::AUTO_STEP);

	//depth = cv::Mat(depthMD.YRes(), depthMD.XRes(), CV_16UC1);
	//memcpy(depth.data, depth_map, depthMD.YRes() * depthMD.XRes() * sizeof(unsigned short));

	mStatus = 2;

	//Capture frames

	if (m_use_capture) {

		//saveRawFrame(colorMD.FrameID(), &colorMD, &depthMD);
		saveFrame(colorMD.FrameID(), rgb, depth);
	}
}

bool XtionStreamReader::saveRawFrame(int frame, xn::ImageMetaData *colorMD, xn::DepthMetaData *depthMD) {

	char path[100] = "";

	sprintf_s(path, "%s\\rgb\\color_map_%d.raw", m_DATA_DIR.c_str(), frame);
	xnOSSaveFile(path, colorMD->Data(), colorMD->DataSize());

	sprintf_s(path, "%s\\depth\\depth_map_%d.raw", m_DATA_DIR.c_str(), frame);
	xnOSSaveFile(path, depthMD->Data(), depthMD->DataSize());

	return true;
}

bool XtionStreamReader::saveFrame(int frame, cv::Mat &rgb, cv::Mat &depth) {

	char path[100] = "";

	sprintf_s(path, "%s/rgb/color_map_%d.png", m_DATA_DIR.c_str(), frame);
	cv::imwrite(path, rgb);

	sprintf_s(path, "%s/depth/depth_map_%d.png", m_DATA_DIR.c_str(), frame);
	cv::imwrite(path, depth);

	return true;
}

//Function to compute the focal length given the field of view angle and the optical center.
float XtionStreamReader::computeFocalLength(float fov_angle, float center) {

	fov_angle *= M_PI / 180; //Angle to radians 

	return center / tanf(fov_angle / 2.0f);
}

//Function to compute the focal length given the field of view in radians and the optical center.
float XtionStreamReader::computeFocalLengthRadians(float fov_angle, float center) {

	return center / tanf(fov_angle / 2.0f);
}

Matrix3f XtionStreamReader::getCameraIntrinsics()
{
	float fx, fy, cx, cy;

	//Optical center
	cx = m_x_res / 2.0f - 0.5f;
	cy = m_y_res / 2.0f - 0.5f;

	/*Focal length from degrees
	fx = computeFocalLength(m_fov_x_degrees, cx);
	fy = computeFocalLength(m_fov_y_degrees, cy);*/

	//Focal length 
	fx = computeFocalLengthRadians(m_fov_x, cx);
	fy = computeFocalLengthRadians(m_fov_y, cy);

	Matrix3f i;
	i << fx, 0, cx,
		0, fy, cy,
		0, 0, 0;

	return i;
}
