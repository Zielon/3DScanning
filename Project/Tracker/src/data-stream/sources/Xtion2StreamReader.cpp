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
	//const char* deviceURI = "\\?\usb#vid_1d27&pid_0601&mi_00#6&833cf63&0&0000#{c3b5f022-5a42-1980-1909-ea72095601b1}";

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

	//openni::VideoStream m_color_stream;

	/*rc = m_color_stream.create(device, openni::SENSOR_COLOR);
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

	openni::VideoStream* pStream = &m_color_stream;
	openni::VideoFrameRef frame;
	int changedStreamDummy;

	rc = openni::OpenNI::waitForAnyStream(&pStream, 1, &changedStreamDummy, READ_WAIT_TIMEOUT);

	if (rc != openni::STATUS_OK)
	{
		printf("Wait failed! (timeout is %d ms)\n%s\n", READ_WAIT_TIMEOUT, openni::OpenNI::getExtendedError());
		return false;
	}

	rc = m_color_stream.readFrame(&frame);

	if (rc != openni::STATUS_OK)
	{
		printf("Read failed!\n%s\n", openni::OpenNI::getExtendedError());
		return -1;
	}

	/*if (frame.getVideoMode().getPixelFormat() != PIXEL_FORMAT_DEPTH_1_MM && frame.getVideoMode().getPixelFormat() != PIXEL_FORMAT_DEPTH_100_UM)
	{
		printf("Unexpected frame format\n");
		return -1;
	}

	openni::RGB888Pixel* pColor = (openni::RGB888Pixel*)frame.getData();

	printf("Color Image Resolution (%d,%d)\n", frame.getHeight(), frame.getWidth());*/

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

	/*if (!depth.isValid() || !color.isValid())
	{
		printf("SimpleViewer: No valid streams. Exiting\n");
		openni::OpenNI::shutdown();
		return 2;
	}*/

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
	openni::VideoFrameRef frame;
	int changedStreamDummy;

	rc = openni::OpenNI::waitForAnyStream(&pStream, 1, &changedStreamDummy, READ_WAIT_TIMEOUT);
	
	if (rc != openni::STATUS_OK)
	{
		printf("Wait failed! (timeout is %d ms)\n%s\n", READ_WAIT_TIMEOUT, openni::OpenNI::getExtendedError());
		return -1;
	}

	rc = m_color_stream.readFrame(&frame);
	
	if (rc != openni::STATUS_OK)
	{
		printf("Read failed!\n%s\n", openni::OpenNI::getExtendedError());
		return -1;
	}

	/*if (frame.getVideoMode().getPixelFormat() != PIXEL_FORMAT_DEPTH_1_MM && frame.getVideoMode().getPixelFormat() != PIXEL_FORMAT_DEPTH_100_UM)
	{
		printf("Unexpected frame format\n");
		return -1;
	}*/

	//Getting raw color map
	openni::RGB888Pixel* pColor = (openni::RGB888Pixel*)frame.getData();

	if (m_use_verbose) {
		printf("Color frame %d: resolution (%d, %d), bytes %d\n", frame.getFrameIndex(), frame.getHeight(), frame.getWidth(), frame.getDataSize());

	}

	//OpenCV color image from raw color map
	rgb = cv::Mat(frame.getHeight(), frame.getWidth(), CV_8UC3, (void*)pColor, cv::Mat::AUTO_STEP);

	return 0;
}

Matrix3f Xtion2StreamReader::getCameraIntrinsics()
{
	Matrix3f i;
	
	i << 517.3, 0, 318.6,
		0, 516.5, 255.3,
		0, 0, 0;

	return i;
}