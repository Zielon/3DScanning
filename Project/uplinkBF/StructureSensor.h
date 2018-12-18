#pragma once

struct vec4uc
{
	unsigned char r, g, b, a;
};

//#define USE_RECALIBRATION
#define SAFE_DELETE_ARRAY(a) { delete [] (a); (a) = NULL; }
#include "Uplink/uplink.h"
//#include "Uplink/desktop-server.h"

//#include "StructureSensorCalibration.h"



struct MyDesktopServer : uplink::Server
{
public:
	MyDesktopServer(const std::string& serviceName, int servicePort, objc_weak uplink::ServerDelegate* serverDelegate)
		: Server(serviceName, servicePort, serverDelegate), m_hasCalibrationData(false), m_hasNewSensorData(false),
		m_feedbackImageData(NULL), m_feedbackWidth(0), m_feedbackHeight(0)
	{
		m_depthBuffer = NULL;
		m_depthBufferF = NULL;
		m_colorBuffer = NULL;
		m_colorBufferRGBX = NULL;

		m_depthBufferMapped = NULL;
		m_colorBufferMapped = NULL;

		// feedback image
		m_feedbackWidth = 0;
		m_feedbackHeight = 0;
		m_feedbackImageData = NULL;//new uplink::uint8[feedbackWidth * feedbackHeight * 3];
		m_feedbackImage.format = uplink::ImageFormat_Empty;
		m_feedbackImage.width = 0;
		m_feedbackImage.height = 0;
	}

	~MyDesktopServer()
	{
		Server::clear();
		SAFE_DELETE_ARRAY(m_depthBuffer);
		SAFE_DELETE_ARRAY(m_colorBuffer);
		SAFE_DELETE_ARRAY(m_depthBufferF);
		SAFE_DELETE_ARRAY(m_colorBufferRGBX);
		SAFE_DELETE_ARRAY(m_depthBufferMapped);
		SAFE_DELETE_ARRAY(m_colorBufferMapped);
		SAFE_DELETE_ARRAY(m_feedbackImageData);
	}

	bool hasCalibration() { return m_hasCalibrationData; }

	void getCalibration(uplink::CameraCalibration& calibrationDepth, uplink::CameraCalibration& calibrationColor, unsigned int& depthWidth, unsigned int& depthHeight, unsigned int& colorWidth, unsigned int& colorHeight)
	{
		calibrationDepth = m_calibrationDepth;
		calibrationColor = m_calibrationColor;
		depthWidth = m_depthWidth;
		depthHeight = m_depthHeight;
		colorWidth = m_colorWidth;
		colorHeight = m_colorHeight;
	}

	void updateCalibration(const uplink::CameraCalibration& calibrationDepth, const uplink::CameraCalibration& calibrationColor, unsigned int depthWidth, unsigned int depthHeight, unsigned int colorWidth, unsigned int colorHeight)
	{
		m_calibrationDepth = calibrationDepth;
		m_calibrationColor = calibrationColor;
		m_depthWidth = depthWidth;
		m_depthHeight = depthHeight;
		m_colorWidth = colorWidth;
		m_colorHeight = colorHeight;
		m_hasCalibrationData = true;
	}

	bool hasNewSensorData()
	{
		return m_hasNewSensorData;
	}

#ifndef USE_RECALIBRATION
	std::pair<float*, UCHAR*> process(float* oldDepth, UCHAR* oldColor)
	{
		if (m_depthBufferF == NULL) m_depthBufferF = new float[m_depthWidth * m_depthHeight];
		if (m_colorBufferRGBX == NULL) m_colorBufferRGBX = new UCHAR[4 * m_colorWidth * m_colorHeight];
		for (unsigned int i = 0; i < m_colorWidth * m_colorHeight; ++i)
		{
			m_colorBufferRGBX[4 * i + 0] = m_colorBuffer[3 * i + 0];
			m_colorBufferRGBX[4 * i + 1] = m_colorBuffer[3 * i + 1];
			m_colorBufferRGBX[4 * i + 2] = m_colorBuffer[3 * i + 2];
			m_colorBufferRGBX[4 * i + 3] = 255;
		}
		for (unsigned int i = 0; i < m_depthWidth * m_depthHeight; ++i)
			m_depthBufferF[i] = m_depthBuffer[i] / 1000.0f;
	
		m_hasNewSensorData = false;
	
		return std::pair<float*, UCHAR*>(m_depthBufferF, m_colorBufferRGBX);
	}
#endif

#ifdef USE_RECALIBRATION
	std::pair<float*, UCHAR*> process(float* oldDepth, UCHAR* oldColor, Calibration* calib)
	{
		if (m_depthBufferF == NULL) m_depthBufferF = new float[m_colorWidth * m_colorHeight];
		if (m_colorBufferRGBX == NULL) m_colorBufferRGBX = new UCHAR[4 * m_colorWidth * m_colorHeight];


		if (m_depthBufferMapped == NULL) m_depthBufferMapped = new USHORT[m_colorWidth * m_colorHeight];
		if (m_colorBufferMapped == NULL) m_colorBufferMapped = new UCHAR[3 * m_colorWidth * m_colorHeight];

		calib->calibrateFrame((vec3uc*)m_colorBuffer, m_depthBuffer, (vec3uc*)m_colorBufferMapped, m_depthBufferMapped);


		for (unsigned int i = 0; i < m_colorWidth * m_colorHeight; ++i)
		{
			m_colorBufferRGBX[4 * i + 0] = m_colorBufferMapped[3 * i + 0];
			m_colorBufferRGBX[4 * i + 1] = m_colorBufferMapped[3 * i + 1];
			m_colorBufferRGBX[4 * i + 2] = m_colorBufferMapped[3 * i + 2];
			m_colorBufferRGBX[4 * i + 3] = 255;
			m_depthBufferF[i] = m_depthBufferMapped[i] / 1000.0f;
		}

		m_hasNewSensorData = false;
	
		return std::pair<float*, UCHAR*>(m_depthBufferF, m_colorBufferRGBX);
	}
#endif

	void receive(USHORT* depthBuffer, UCHAR* colorBuffer)
	{
		if (!m_hasCalibrationData) return;

		// std::cout << "m_depthWidth:\t" <<  m_depthWidth << std::endl;
		// std::cout << "m_depthHeight:\t" << m_depthHeight << std::endl;
		// std::cout << "m_colorWidth:\t" << m_colorWidth << std::endl;
		// std::cout << "m_colorHeight:\t" << m_colorHeight << std::endl;

		if (m_depthBuffer == NULL) m_depthBuffer = new USHORT[m_depthWidth * m_depthHeight];
		if (m_colorBuffer == NULL) m_colorBuffer = new UCHAR[3*m_colorWidth * m_colorHeight];

		memcpy(m_depthBuffer, depthBuffer, sizeof(USHORT)*m_depthWidth * m_depthHeight);
		memcpy(m_colorBuffer, colorBuffer, sizeof(UCHAR) * 3 * m_colorWidth * m_colorHeight);

		m_hasNewSensorData = true;
	}



	uplink::Image& getFeedbackImage()
	{
		return m_feedbackImage;
	}

	void updateFeedbackImage(BYTE* dataRGBA, unsigned int w, unsigned int h)
	{
		if (w != m_feedbackWidth || h != m_feedbackHeight)
		{
			SAFE_DELETE_ARRAY(m_feedbackImageData);
			m_feedbackWidth = w;
			m_feedbackHeight = h;
			m_feedbackImageData = new unsigned char[3 * w*h];
		}

		m_feedbackImage.format = uplink::ImageFormat_RGB;
		m_feedbackImage.width = m_feedbackWidth;
		m_feedbackImage.height = m_feedbackHeight;
		m_feedbackImage.planes[0].buffer = m_feedbackImageData;
		m_feedbackImage.planes[0].sizeInBytes = sizeof(unsigned char) * m_feedbackWidth * m_feedbackHeight;
		m_feedbackImage.planes[0].bytesPerRow = sizeof(unsigned char) * m_feedbackWidth;

#pragma omp parallel for
		for (int i = 0; i < m_feedbackImage.width * m_feedbackImage.height; i++) {
			m_feedbackImageData[i * 3 + 0] = dataRGBA[i * 4 + 0];
			m_feedbackImageData[i * 3 + 1] = dataRGBA[i * 4 + 1];
			m_feedbackImageData[i * 3 + 2] = dataRGBA[i * 4 + 2];
		}
	}

private:
	bool m_hasCalibrationData;

	uplink::CameraCalibration m_calibrationDepth, m_calibrationColor;
	unsigned int m_depthWidth, m_depthHeight, m_colorWidth, m_colorHeight;

	USHORT* m_depthBuffer;
	float* m_depthBufferF;
	UCHAR* m_colorBuffer;
	UCHAR* m_colorBufferRGBX;

	USHORT* m_depthBufferMapped;
	UCHAR* m_colorBufferMapped;

	bool m_hasNewSensorData;

	uplink::Image m_feedbackImage; // sent back to app
	/*uint8*/unsigned char* m_feedbackImageData;
	unsigned int m_feedbackWidth, m_feedbackHeight;
};

//------------------------------------------------------------------------------

struct ExampleServerSession : uplink::DesktopServerSession
{
	ExampleServerSession(int socketDescriptor, MyDesktopServer* server)
		: DesktopServerSession(socketDescriptor, server), m_server(server)
	{
	}


	void toggleExposureAndWhiteBalance(bool lock = false) //lock forces the lock
	{
		uplink::SessionSetup sessionSetup;

		static bool toggle = true;

		if (toggle || lock)
		{
			sessionSetup.addSetColorCameraExposureModeAction(uplink::ColorCameraExposureMode_Locked);
			sessionSetup.addSetColorCameraWhiteBalanceModeAction(uplink::ColorCameraWhiteBalanceMode_Locked);

			std::cout << "awb/exp LOCKED" << std::endl;
			//uplink_log_info("Locked exposure and white balance.");
		}
		else
		{
			sessionSetup.addSetColorCameraExposureModeAction(uplink::ColorCameraExposureMode_ContinuousAuto);
			sessionSetup.addSetColorCameraWhiteBalanceModeAction(uplink::ColorCameraWhiteBalanceMode_ContinuousAuto);

			std::cout << "awb/exp unlocked" << std::endl;
			//uplink_log_info("Automatic exposure and white balance.");
		}

		server()._currentSession->sendSessionSetup(sessionSetup);

		toggle = !toggle;
	}

	virtual void onCustomCommand(const std::string& command)
	{
		// FIXME: Implement.
	}

	virtual bool onMessage(const uplink::Message& message)
	{
		switch (message.kind())
		{
		case uplink::MessageKind_DeviceMotionEvent:
			{
				std::cout << "IMU" << std::endl;
				break;
			}

		case uplink::MessageKind_CameraFrame:
		{

			const uplink::CameraFrame& cameraFrame = message.as<uplink::CameraFrame>();

			UCHAR* colorBuffer = NULL;
			if (!cameraFrame.colorImage.isEmpty())
			{
				colorBuffer = (UCHAR*)cameraFrame.colorImage.planes[0].buffer;
			}

			USHORT* depthBuffer = (USHORT*)cameraFrame.depthImage.planes[0].buffer;
			int     depthWidth  = int(cameraFrame.depthImage.width);
			int     depthHeight = int(cameraFrame.depthImage.height);
			if (!cameraFrame.depthImage.isEmpty())
			{
				// Convert shifts to depth values.
				uplink::shift2depth(depthBuffer, depthWidth * depthHeight);
			}

			static bool storeCalibration = true;
			static unsigned long long count = 1; // FIXME: Use a real steady-rate timer.
			if (!cameraFrame.colorImage.isEmpty() && !cameraFrame.depthImage.isEmpty()) // valid
			{
				if (storeCalibration)
				{
					if (colorBuffer != NULL) // wait for first color frame
					{				
						m_server->updateCalibration(
							cameraFrame.depthImage.cameraInfo.calibration,
							cameraFrame.colorImage.cameraInfo.calibration,
							(unsigned int)cameraFrame.depthImage.width,
							(unsigned int)cameraFrame.depthImage.height,
							(unsigned int)cameraFrame.colorImage.width,
							(unsigned int)cameraFrame.colorImage.height);
						//server().ui().setPingPongColorCameraInfo(cameraFrame.colorImage.cameraInfo);
						storeCalibration = false;
					}
				}

				m_server->receive(depthBuffer, colorBuffer);
				count++;
			}

			// Send ping-pong feedback image.
			const uplink::Image& feedback = m_server->getFeedbackImage();
			if (!feedback.isEmpty()) {
				sendImage(const_cast<uplink::Image&>(feedback));
			}


			//if (dumpStatsPeriodically && 0 == count % 150)
			//{
			//    uplink_log_info("Camera receiving rate: %f Hz", server()._currentSession->messageStats[MessageKind_CameraFrame].receiving.rate.windowedRate());
			//    uplink_log_info("Motion receiving rate: %f Hz", server()._currentSession->messageStats[MessageKind_DeviceMotionEvent].receiving.rate.windowedRate());
			//    uplink_log_info("Feedback image sending rate: %f Hz", server()._currentSession->channels[ChannelId_Feedback].stats.sending.rate.windowedRate());
			//}

			++count;

			break;
		}

		default:
			{
				std::cout << "Other" << std::endl;
				break;
			}
		}

		return true;
	}


	//static FrameTimer s_timer;
	MyDesktopServer* m_server;
};

//------------------------------------------------------------------------------

struct ExampleSessionSetup : uplink::SessionSetup
{
	ExampleSessionSetup()
	{
		//addSetColorModeAction(uplink::ColorMode_VGA);
		addSetColorModeAction(uplink::ColorMode_1296x968);
		addSetDepthModeAction(uplink::DepthMode_VGA);
#ifdef USE_RECALIBRATION
		addSetRegistrationModeAction(uplink::RegistrationMode_None);
#else
		addSetRegistrationModeAction(uplink::RegistrationMode_RegisteredDepth);
#endif
		//addSetFrameSyncModeAction(uplink::FrameSyncMode_Depth);

		addSetSporadicFrameColorAction(false);
		addSetSporadicFrameColorDivisorAction(1);

		uplink::ChannelSettings channelSettings;
		channelSettings.droppingStrategy = uplink::DroppingStrategy_RandomOne;
		channelSettings.droppingThreshold = 90;
		channelSettings.bufferingStrategy = uplink::BufferingStrategy_Some;

		addSetRGBDFrameChannelAction(channelSettings);

		addSetSendMotionAction(false);
		addSetMotionRateAction(100);

		addSetColorCameraExposureModeAction(uplink::ColorCameraExposureMode_Locked);
		addSetColorCameraWhiteBalanceModeAction(uplink::ColorCameraWhiteBalanceMode_Locked);

		addSetDepthCameraCodecAction(uplink::ImageCodecId_CompressedShifts);
		addSetColorCameraCodecAction(uplink::ImageCodecId_JPEG);
		//addSetFeedbackImageCodecAction(uplink::ImageCodecId_JPEG);

	}
};

//------------------------------------------------------------------------------

struct ExampleServerDelegate : uplink::ServerDelegate
{
	virtual uplink::ServerSession* newSession(int socketDescriptor, uplink::Server* server)
	{
		_server = (MyDesktopServer*)server;

		return new ExampleServerSession(socketDescriptor, _server);
	}

	virtual void onConnect(uintptr_t sessionId)
	{
		_server->_currentSession->sendSessionSetup(
			ExampleSessionSetup()
			//SporadicColorSessionSetup()
			//Depth60FPSSessionSetup()
			//WXGASessionSetup()
			);
	}

	MyDesktopServer* _server;
};




//------------------------------------------------------------------------------

class StructureSensor
{
public:
	StructureSensor(bool nearMode = true);
	~StructureSensor();


	//! Processes the next frame data
	HRESULT ProcessNextFrame();

	//! Toggles the Kinect to near-mode; default is far mode
	HRESULT ToggleNearMode() { return E_FAIL; }

	//! Toggle enable auto white balance
	HRESULT ToggleAutoWhiteBalance(bool forceLock = false)
	{
		((ExampleServerSession*)m_server._currentSession)->toggleExposureAndWhiteBalance(forceLock);
		return S_OK;
	}


	void startReceivingFrames() { m_server.start(); }
	void stopReceivingFrames() { m_server.stop(); }

	void updateFeedbackImage(BYTE* dataRGBA, unsigned int w, unsigned int h)
	{
		m_server.updateFeedbackImage(dataRGBA, w, h);
	}

	virtual std::string GetSensorName() { return "StructureIO"; }

	HRESULT CreateFirstConnected();

private:

	void waitForConnection();

	ExampleServerDelegate m_serverDelegate;
	MyDesktopServer m_server;

	float* m_oldDepth;
	UCHAR* m_oldColor;

	USHORT m_minDepth, m_maxDepth; 
	bool m_hasPlayerID, m_hasFloatValues, m_needsRegistration; 

#ifdef USE_RECALIBRATION
	Calibration* m_pCalibration;
#endif

	unsigned int depthWidthIn, depthHeightIn, colorWidthIn, colorHeightIn;

	float* m_depthFloat;
	float* m_colorRGBX;

	int   m_depthImageWidth;
	int   m_depthImageHeight;

	int   m_colorImageWidth;
	int   m_colorImageHeight;

};
