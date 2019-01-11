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

	openni::Device device;
	openni::VideoStream color, depth;

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

	std::cin.get();

	//Open device (sensor)
	rc = device.open(deviceURI);

	std::cin.get();

	if (rc != openni::STATUS_OK)
	{
		printf("Device open failed:\n%s\n", openni::OpenNI::getExtendedError());
		openni::OpenNI::shutdown();
		std::cin.get();
		return false;
	}

	//Create Color Stream
	rc = color.create(device, openni::SENSOR_COLOR);
	if (rc == openni::STATUS_OK)
	{
		rc = color.start();

		if (rc != openni::STATUS_OK)
		{
			printf("Couldn't start color stream:\n%s\n", openni::OpenNI::getExtendedError());
			color.destroy();
		}
	}
	else
	{
		printf("SimpleViewer: Couldn't find color stream:\n%s\n", openni::OpenNI::getExtendedError());
	}

	/*if (!depth.isValid() || !color.isValid())
	{
		printf("SimpleViewer: No valid streams. Exiting\n");
		openni::OpenNI::shutdown();
		return 2;
	}*/

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

	return false;
}

bool Xtion2StreamReader::stopReading() {

	return true;
}

bool Xtion2StreamReader::isRunning() {
	return true;
}

int Xtion2StreamReader::readFrame(cv::Mat &rgb, cv::Mat &depth) {

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