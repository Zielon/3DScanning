#include "StructureSensor.h"
#include <iomanip>

namespace uplink {
	Context context;
}


StructureSensor::StructureSensor(bool nearMode) : 
	m_server("UplinkTool", 6666, &m_serverDelegate) 
{
	m_minDepth = (USHORT)(200);
	m_maxDepth = (USHORT)(1300); // 1,3m //(USHORT)(4000);
	m_hasPlayerID = false;
	m_hasFloatValues = true;
	m_needsRegistration = false;

	//m_server.init(20, GlobalAppState::get().s_windowWidth, GlobalAppState::get().s_windowHeight); // depth/color buffer size //!!! TODO
	m_oldDepth = NULL;
	m_oldColor = NULL;
#ifdef USE_RECALIBRATION
	m_pCalibration = NULL;
#endif
}

StructureSensor::~StructureSensor()
{
	if (!m_server.isStopped()) m_server.stop();
}

void StructureSensor::waitForConnection()
{
	std::cout << "waiting for connection... ";
	m_server.startListening();

	while (!m_server.hasCalibration()) {
		// wait for calibration
		Sleep(10);
		//std::cout << "wait for calibration..." << std::endl;
	}

	std::cout << "ready!" << std::endl;
}


HRESULT StructureSensor::ProcessNextFrame()
{
#ifdef USE_RECALIBRATION
	std::pair<float*, UCHAR*> frames = m_server.process(m_oldDepth, m_oldColor, m_pCalibration);
#else
	std::pair<float*, UCHAR*> frames = m_server.process(m_oldDepth, m_oldColor);
#endif
	if (frames.first == NULL || frames.second == NULL)
	{
		std::cout << "structure sensor unable to process frame" << std::endl;
		return E_FAIL;
	}

	// depth
	memcpy(m_depthFloat, frames.first, sizeof(float)*m_depthImageWidth*m_depthImageHeight);
	for (int y = 0; y < m_depthImageHeight; ++y)
	{
		for (int x = 0; x < m_colorImageWidth; ++x)
		{
			//m_depthF[y*m_depthImageWidth + x] *= 8;
			//m_depthF[y*m_depthImageWidth + x] = frames.first[y*m_depthImageWidth + x];
			if (m_depthFloat[y*m_depthImageWidth + x] > float(m_maxDepth)*0.001f || m_depthFloat[y*m_depthImageWidth + x] < float(m_minDepth)*0.001f)
				m_depthFloat[y*m_depthImageWidth + x] = FLT_MIN;
		}
	}

	// color
	//memcpy(m_colorRGBX, (BYTE*)frames.second, 4*sizeof(BYTE)*m_colorImageWidth*m_colorImageHeight);
	unsigned int offsetX = (m_colorImageWidth - colorWidthIn) / 2;
	for (int y = 0; y < /*m_colorImageHeight*/colorHeightIn; ++y)
	{
		for (int x = 0; x < m_colorImageWidth; ++x)
		{
			m_colorRGBX[4 * (y*m_colorImageWidth + x) + 0] = 255;
			m_colorRGBX[4 * (y*m_colorImageWidth + x) + 1] = 0;
			m_colorRGBX[4 * (y*m_colorImageWidth + x) + 2] = 0;
			m_colorRGBX[4 * (y*m_colorImageWidth + x) + 3] = 255;
		}
		memcpy(&(m_colorRGBX[4 * (y*m_colorImageWidth + offsetX)]), &(frames.second[4 * y*colorWidthIn]), 4 * sizeof(BYTE) * colorWidthIn);
	}


	m_oldDepth = frames.first;
	m_oldColor = frames.second;
	return S_OK;
}



HRESULT StructureSensor::CreateFirstConnected()
{
	std::cout << "createFirstConnected" << std::endl;
#ifdef USE_RECALIBRATION
	//m_pCalibration = new Calibration("Matthias One.txt", "Matthias One.lut");
	//m_pCalibration = new Calibration("matthias_one_new.txt", "matthias_one_new.lut");
	m_pCalibration = new Calibration("matthias_one_v2.txt", "matthias_one_v2.lut");
#endif


	waitForConnection();

	// get calibration
	uplink::CameraCalibration calibrationDepth, calibrationColor;
	//unsigned int depthWidth, depthHeight, colorWidth, colorHeight;
	m_server.getCalibration(calibrationDepth, calibrationColor, depthWidthIn, depthHeightIn, colorWidthIn, colorHeightIn);
	// RGBDSensor::init(depthWidth, depthHeight, colorWidth, colorHeight, 1);
	// 
	// mat4f depthExtrinsics = quatf(calibrationDepth.qw, calibrationDepth.qx, calibrationDepth.qy, calibrationDepth.qz).matrix4x4(); // rotation
	// depthExtrinsics.setTranslation(vec3f(calibrationDepth.tx, calibrationDepth.ty, calibrationDepth.tz));
	// RGBDSensor::initializeDepthIntrinsics(calibrationDepth.fx, calibrationDepth.fy, calibrationDepth.cx, calibrationDepth.cy);
	// RGBDSensor::initializeDepthExtrinsics(depthExtrinsics);
	// 
	// mat4f colorExtrinsics = quatf(calibrationColor.qw, calibrationColor.qx, calibrationColor.qy, calibrationColor.qz).matrix4x4(); // rotation
	// colorExtrinsics.setTranslation(vec3f(calibrationColor.tx, calibrationColor.ty, calibrationColor.tz));
	// RGBDSensor::initializeColorIntrinsics(calibrationColor.fx, calibrationColor.fy, calibrationColor.cx, calibrationColor.cy);
	// RGBDSensor::initializeColorExtrinsics(mat4f::identity());

#ifndef USE_RECALIBRATION

	m_depthImageWidth = depthWidthIn;
	m_depthImageHeight = depthHeightIn;
	m_colorImageWidth = ceil(colorWidthIn/32.0)*32;
	m_colorImageHeight = colorHeightIn;
	//m_colorImageHeight = ceil(colorHeightIn / 32.0) * 32;;

	float focalLengthX = calibrationDepth.fx;
	float focalLengthY = calibrationDepth.fy;
	float cX = calibrationDepth.cx;
	float cY = calibrationDepth.cy;

	cX += (m_colorImageWidth - colorWidthIn) / 2;


	/*/ //TODO: Calibration
	InitializeIntrinsics(focalLengthX, focalLengthY, cX, cY);

	Eigen::Matrix4f depthExtrinsics;
	{
		depthExtrinsics.setIdentity();
		Eigen::Quaternionf q;
		q.x() = calibrationDepth.qx;
		q.y() = calibrationDepth.qy;
		q.z() = calibrationDepth.qz;
		depthExtrinsics.block<3, 3>(0, 0) = q.toRotationMatrix();
		depthExtrinsics.block<3, 1>(0, 3) = Vector3f(calibrationDepth.tx, calibrationDepth.ty, calibrationDepth.tz);
	}

	Eigen::Matrix4f colorExtrinsics;
	{
		colorExtrinsics.setIdentity();
		Eigen::Quaternionf q;
		q.x() = calibrationColor.qx;
		q.y() = calibrationColor.qy;
		q.z() = calibrationColor.qz;
		colorExtrinsics.block<3, 3>(0, 0) = q.toRotationMatrix();
		colorExtrinsics.block<3, 1>(0, 3) = Vector3f(calibrationColor.tx, calibrationColor.ty, calibrationColor.tz);
	}

	std::cout << "depthIntrinsics:\t" << calibrationDepth.fx << " " << calibrationDepth.fy << " " << calibrationDepth.cx << " " << calibrationDepth.cy << std::endl;
	std::cout << "colorIntrinsics:\t" << calibrationColor.fx << " " << calibrationColor.fy << " " << calibrationColor.cx << " " << calibrationColor.cy << std::endl;
	std::cout << "depthExtrinsics:\n" << depthExtrinsics << std::endl;
	std::cout << "colorExtrinsics:\n" << colorExtrinsics << std::endl;
	*/
#endif


#ifdef USE_RECALIBRATION
	if (m_pCalibration->getWidth() != colorWidthIn || m_pCalibration->getHeight() != colorHeightIn)
	{
		std::cout << "[StructureSensor] ERROR calibration does not match the input!" << std::endl;
		return E_FAIL;
	}

	RGBDSensor::init(colorWidthIn, colorHeightIn, colorWidthIn, colorHeightIn, 1);

	m_depthIntrinsics = m_pCalibration->getIntrinsics();
	m_colorIntrinsics = m_pCalibration->getIntrinsics();
	m_depthIntrinsicsInv = m_depthIntrinsics.getInverse();
	m_colorIntrinsicsInv = m_colorIntrinsics.getInverse();
	m_depthExtrinsics.setIdentity();
	m_colorExtrinsics.setIdentity();
	m_depthExtrinsicsInv.setIdentity();
	m_colorExtrinsicsInv.setIdentity();

	std::cout << "depth intrinsics: " << std::endl << m_depthIntrinsics << std::endl;
	std::cout << "color intrinsics: " << std::endl << m_colorIntrinsics << std::endl;
	std::cout << "depth extrinsics: " << std::endl << m_depthExtrinsics << std::endl;
	std::cout << "color extrinsics: " << std::endl << m_colorExtrinsics << std::endl;
#endif

	return S_OK;
}


