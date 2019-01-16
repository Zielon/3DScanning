#include "../Headers/Xtion2StreamReader.h"

Xtion2StreamReader::Xtion2StreamReader(bool realtime, bool verbose, bool capture) {

	m_realtime = realtime;
	m_use_capture = capture;
	m_use_verbose = verbose;
}

Xtion2StreamReader::~Xtion2StreamReader() {

	//Release resources
}

bool Xtion2StreamReader::initContext() {

	openni::Status rc = openni::STATUS_OK;

	//openni::Device device;

	const char* deviceURI = openni::ANY_DEVICE;

	printf("Device URI:\n%s\n", deviceURI);

	//Initialize OpenNI
	rc = openni::OpenNI::initialize();

	if (rc != openni::STATUS_OK)
	{
		printf("Initialize failed\n%s\n", openni::OpenNI::getExtendedError());
		return false;
	}

	printf("After initialization:\n%s\n", openni::OpenNI::getExtendedError());

	openni::Array<openni::DeviceInfo> deviceList;
	openni::OpenNI::enumerateDevices(&deviceList);
	
	printf("List of devices\n");
	for (int i = 0; i < deviceList.getSize(); ++i)
	{
		printf("Device \"%s\" already connected\n", deviceList[i].getUri());
	}

	//Open device (sensor)
	rc = m_device.open(deviceURI);

	if (rc != openni::STATUS_OK)
	{
		printf("Device open failed:\n%s\n", openni::OpenNI::getExtendedError());
		openni::OpenNI::shutdown();
		std::cin.get();
		return false;
	}

	//printf("Stream created properly");

	return true;
}

bool Xtion2StreamReader::nextFrameAvailable() {

	return true;
}

int Xtion2StreamReader::getSequentialFrame(cv::Mat &rgb, cv::Mat &depth) {

	return readFrame(rgb, depth);
}

int Xtion2StreamReader::getLatestFrame(cv::Mat &rgb, cv::Mat &depth) {

	return readFrame(rgb, depth);
}

bool Xtion2StreamReader::startReading() {

	openni::Status rc = openni::STATUS_OK;

	if (!this->initContext()) {
		printf("Failed to create input stream context\n");
		return false;
	}

	//Create Color Stream
	rc = m_color_stream.create(m_device, openni::SENSOR_COLOR);
	if (rc == openni::STATUS_OK)
	{
		rc = m_color_stream.start();

		if (rc != openni::STATUS_OK)
		{
			printf("Couldn't start color stream:\n%s\n", openni::OpenNI::getExtendedError());
			m_color_stream.destroy();
		}
	}
	else
	{
		printf("Couldn't find color stream:\n%s\n", openni::OpenNI::getExtendedError());
	}

	if (!m_color_stream.isValid())
	{
		printf("No valid streams. Exiting\n");
		openni::OpenNI::shutdown();

		return false;
	}

	//Create Depth Stream

	rc = m_depth_stream.create(m_device, openni::SENSOR_DEPTH);
	if (rc == openni::STATUS_OK)
	{
		rc = m_depth_stream.start();
		if (rc != openni::STATUS_OK)
		{
			printf("Couldn't start depth stream:\n%s\n", openni::OpenNI::getExtendedError());
			m_depth_stream.destroy();
		}
	}
	else
	{
		printf("Couldn't find depth stream:\n%s\n", openni::OpenNI::getExtendedError());
	}

	//Validate streams
	if (!m_color_stream.isValid() || !m_depth_stream.isValid())
	{
		printf("No valid streams. Exiting\n");
		openni::OpenNI::shutdown();
		return 2;
	}

	// Obtainining first color frame in order to obtain maps resolution
	openni::VideoStream* pStream = &m_color_stream;
	openni::VideoFrameRef colorFrame;
	int changedStreamDummy;

	rc = openni::OpenNI::waitForAnyStream(&pStream, 1, &changedStreamDummy, READ_WAIT_TIMEOUT);

	if (rc != openni::STATUS_OK)
	{
		printf("Wait failed! (timeout is %d ms)\n%s\n", READ_WAIT_TIMEOUT, openni::OpenNI::getExtendedError());
		return -1;
	}

	//Read  first color frame to obtain dimensions
	rc = m_color_stream.readFrame(&colorFrame);

	if (rc != openni::STATUS_OK)
	{
		printf("Read failed!\n%s\n", openni::OpenNI::getExtendedError());
		return -1;
	}

	//Validate color format
	if (colorFrame.getVideoMode().getPixelFormat() != openni::PIXEL_FORMAT_RGB888)
	{
		printf("Unexpected frame format\n");
		return -1;
	}

	//Setting sensor intrinsics
	m_height_rgb = colorFrame.getHeight();
	m_width_rgb = colorFrame.getWidth();
	m_fov_x = m_color_stream.getHorizontalFieldOfView();
	m_fov_y = m_color_stream.getVerticalFieldOfView();
	m_depth_stream.getMaxPixelValue();//10000

	return true;
}

bool Xtion2StreamReader::stopReading() {

	return true;
}

bool Xtion2StreamReader::isRunning() {
	return true;
}

int Xtion2StreamReader::readFrame(cv::Mat &rgb, cv::Mat &depth) {

	openni::Status rc = openni::STATUS_OK;
	openni::VideoStream* pStream = &m_color_stream;
	openni::VideoFrameRef colorFrame, depthFrame;
	int changedStreamDummy;

	rc = openni::OpenNI::waitForAnyStream(&pStream, 1, &changedStreamDummy, READ_WAIT_TIMEOUT);
	
	if (rc != openni::STATUS_OK)
	{
		printf("Wait failed! (timeout is %d ms)\n%s\n", READ_WAIT_TIMEOUT, openni::OpenNI::getExtendedError());
		return -1;
	}

	//Read color frame
	rc = m_color_stream.readFrame(&colorFrame);
	
	if (rc != openni::STATUS_OK)
	{
		printf("Read failed!\n%s\n", openni::OpenNI::getExtendedError());
		return -1;
	}

	//Validate color format
	if (colorFrame.getVideoMode().getPixelFormat() != openni::PIXEL_FORMAT_RGB888)
	{
		printf("Unexpected frame format\n");
		return -1;
	}

	openni::RGB888Pixel* pColor = (openni::RGB888Pixel*)colorFrame.getData();//Getting raw map

	rc = m_depth_stream.readFrame(&depthFrame);

	if (rc != openni::STATUS_OK)
	{
		printf("Read failed!\n%s\n", openni::OpenNI::getExtendedError());
		return -1;
	}

	//Valide depth format
	if (depthFrame.getVideoMode().getPixelFormat() != openni::PIXEL_FORMAT_DEPTH_1_MM && 
		depthFrame.getVideoMode().getPixelFormat() != openni::PIXEL_FORMAT_DEPTH_100_UM)
	{
		printf("Unexpected frame format\n");
		return -1;
	}

	openni::DepthPixel* pDepth = (openni::DepthPixel*)depthFrame.getData();

	if (m_use_verbose) {
		printf("Color frame %d: resolution (%d, %d), bytes %d\n", colorFrame.getFrameIndex(), colorFrame.getHeight(), colorFrame.getWidth(), colorFrame.getDataSize());
		printf("Color frame %d: resolution (%d, %d), bytes %d\n", depthFrame.getFrameIndex(), depthFrame.getHeight(), depthFrame.getWidth(), depthFrame.getDataSize());
	}

	//OpenCV color image from raw color map
	rgb = cv::Mat(colorFrame.getHeight(), colorFrame.getWidth(), CV_8UC3, (void*)pColor, cv::Mat::AUTO_STEP);
	depth = cv::Mat(depthFrame.getHeight(), depthFrame.getWidth(), CV_16UC1, (void*)pDepth, cv::Mat::AUTO_STEP);

	// depth images are stored in milimeters (see http://qianyi.info/scenedata.html )
	depth.convertTo(depth, CV_32FC1, 1.0 / 1000.0);//Right format is CV_16FC1

	//Capture frames
	if (m_use_capture) {

		//saveRawFrame(colorMD.FrameID(), &colorMD, &depthMD);
		saveFrame(colorFrame.getFrameIndex(), rgb, depth);
	}

	return 0;
}

//Function to compute the focal length given the field of view in radians and the optical center.
float Xtion2StreamReader::computeFocalLengthRadians(float fov_angle, float center) {

	return center / tanf(fov_angle / 2.0f);
}

Matrix3f Xtion2StreamReader::getCameraIntrinsics()
{
	float fx, fy, cx, cy;

	//Optical center
	cx = m_width_rgb / 2.0f - 0.5f;
	cy = m_height_rgb / 2.0f - 0.5f;

	//Focal length 
	fx = computeFocalLengthRadians(m_fov_x, cx);
	fy = computeFocalLengthRadians(m_fov_y, cy);

	Matrix3f i;
	i << fx, 0, cx,
		0, fy, cy,
		0, 0, 0;

	return i;
}

bool Xtion2StreamReader::saveFrame(int frame, cv::Mat &rgb, cv::Mat &depth) {

	char path[100] = "";

	sprintf_s(path, "%s/rgb/color_map_%d.png", m_DATA_DIR.c_str(), frame);
	cv::imwrite(path, rgb);

	sprintf_s(path, "%s/depth/depth_map_%d.png", m_DATA_DIR.c_str(), frame);
	cv::imwrite(path, depth);

	return true;
}