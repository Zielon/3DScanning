// This file is part of Uplink, an easy-to-use cross-platform live RGBD streaming protocol.
// Copyright (c) 2016, Occipital, Inc.  All rights reserved.
// License: See LICENSE.

#include <uplink.h>
#include <sstream>
#include "desktop-server.h"
#include "desktop-ui.h"

#include "SniffedSettings.h"




static const bool sendPingPongColorFeedback = true;
static const bool dumpStatsPeriodically = true;

using namespace uplink;


struct ScanectSessionSetup : SessionSetup
{


	ScanectSessionSetup()
	{
		sendHackyByteArray = true; 
	}

	virtual bool serializeWith(Serializer &serializer)
	{
		if (serializer.isWriter())
		{
			//return_false_unless(serializer.asWriter().write(uint16(160)));
			return_false_unless(serializer.asWriter().writeBytes(HACK, ARRAYSIZE(HACK)));
			//::cout <<"cfg\n" <<  SKANECT_CONFIG << "\nEND cfg"<<std::endl; 
		}
	}

};

struct RGBDStreamSessionSetup : SessionSetup
{
	RGBDStreamSessionSetup()
	{
		
		addSetColorModeAction(ColorMode_VGA);
		
		
		addSetDepthModeAction(DepthMode_VGA);
		addSetRegistrationModeAction(RegistrationMode_RegisteredDepth);
		addSetFrameSyncModeAction(FrameSyncMode_Depth);
		

		/* LEGACAY ?
		//addSetColorSendingStrategyAction(ColorSendingStrategy_AllFrames);
		//addSetColorSendingSporadicityAction(1);

		//addSetCameraChannelSettingsAction(ChannelSettings(DroppingStrategy_DropAllButLatestOne));

		//addSetMotionModeAction(MotionMode_None);
		//addSetMotionRateAction(100.f);

		//addSetColorCameraExposureModeAction(ColorCameraExposureMode_ContinuousAuto);
		//addSetColorCameraWhiteBalanceModeAction(ColorCameraWhiteBalanceMode_ContinuousAuto);

		//addSetDepthCameraCodecAction(ImageCodecId_CompressedShifts);
		//addSetColorCameraCodecAction(ImageCodecId_JPEG);
		//addSetColorFeedbackCodecAction(ImageCodecId_JPEG);
		*/

		
		ChannelSettings channelSettings;
		channelSettings.droppingStrategy = DroppingStrategy_RandomOne;
		channelSettings.droppingThreshold = 90;
		channelSettings.bufferingStrategy = BufferingStrategy_Some;

		addSetRGBDFrameChannelAction(channelSettings);

		addSetSendMotionAction(true);
		addSetMotionRateAction(100);

		addSetColorCameraExposureModeAction(ColorCameraExposureMode_ContinuousAuto);
		addSetColorCameraWhiteBalanceModeAction(ColorCameraWhiteBalanceMode_ContinuousAuto);

		addSetDepthCameraCodecAction(ImageCodecId_CompressedShifts);
		addSetColorCameraCodecAction(ImageCodecId_JPEG);
		addSetFeedbackImageCodecAction(ImageCodecId_JPEG);
		
	}
};

//Does the fucking same as above
void configSession(SessionSetup& setup)
{
	setup.addSetColorModeAction(ColorMode_VGA);

	setup.addSetDepthModeAction(DepthMode_VGA);
	setup.addSetRegistrationModeAction(RegistrationMode_RegisteredDepth);
	setup.addSetFrameSyncModeAction(FrameSyncMode_Depth);
	//setup.addSetColorSendingStrategyAction(ColorSendingStrategy_AllFrames);
	//setup.addSetColorSendingSporadicityAction(1);

	//setup.addSetCameraChannelSettingsAction(ChannelSettings(DroppingStrategy_DropAllButLatestOne));

	//setup.addSetMotionModeAction(MotionMode_None);
	//setup.addSetMotionRateAction(100.f);

	//setup.addSetColorCameraExposureModeAction(ColorCameraExposureMode_ContinuousAuto);
	//setup.addSetColorCameraWhiteBalanceModeAction(ColorCameraWhiteBalanceMode_ContinuousAuto);

	//setup.addSetDepthCameraCodecAction(ImageCodecId_CompressedShifts);
	//setup.addSetColorCameraCodecAction(ImageCodecId_JPEG);
	//setup.addSetColorFeedbackCodecAction(ImageCodecId_JPEG);

	ChannelSettings channelSettings;
	channelSettings.droppingStrategy = DroppingStrategy_RandomOne;
	channelSettings.droppingThreshold = 90;
	channelSettings.bufferingStrategy = BufferingStrategy_Some;

	setup.addSetRGBDFrameChannelAction(channelSettings);

	setup.addSetSendMotionAction(true);
	setup.addSetMotionRateAction(100);

	setup.addSetColorCameraExposureModeAction(ColorCameraExposureMode_ContinuousAuto);
	setup.addSetColorCameraWhiteBalanceModeAction(ColorCameraWhiteBalanceMode_ContinuousAuto);

	setup.addSetDepthCameraCodecAction(ImageCodecId_CompressedShifts);
	setup.addSetColorCameraCodecAction(ImageCodecId_JPEG);
	setup.addSetFeedbackImageCodecAction(ImageCodecId_JPEG);
}

//------------------------------------------------------------------------------

struct UplinkSession : DesktopServerSession
{
	UplinkSession(int socketDescriptor, Server* server)
		: DesktopServerSession(socketDescriptor, server)
	{

	}

	virtual void onCustomCommand(const String& command)
	{
		if (command == "RecordButtonPressed")
		{
			std::cout << "Start Recording Pressed!" << std::endl;

		}
	}

	virtual bool onMessage(const Message& message)
	{
		// Do not call blocking functions from this point on. Network performance will suffer greatly otherwise.
		switch (message.kind())
		{
			//case uplink::MessageKind_MessageFragment:
			//{
			//    const MessageFragment& messageFragment = message.as<MessageFragment>();

			//    std::cout << messageFragment.toString() << std::endl;
			//
			//    break;
			//}

		case MessageKind_DeviceMotionEvent:
		{
			std::cout << "IMU" << std::endl;
			break;
		}

		case MessageKind_CameraFrame:
		{
			std::cout << "Jesus fuck, i finally got a frame" << std::endl; 
			const CameraFrame& cameraFrame = message.as<CameraFrame>();

			if (!cameraFrame.colorImage.isEmpty())
			{
				server().ui().setColorImage(
					(const uint8*)cameraFrame.colorImage.planes[0].buffer,
					int(cameraFrame.colorImage.width),
					int(cameraFrame.colorImage.height)
				);
			}


			if (!cameraFrame.depthImage.isEmpty())
			{
				uint16* depthBuffer = (uint16*)cameraFrame.depthImage.planes[0].buffer;
				int     depthWidth = int(cameraFrame.depthImage.width);
				int     depthHeight = int(cameraFrame.depthImage.height);

				// Convert shifts to depth values.
				shift2depth(depthBuffer, depthWidth * depthHeight);

				server().ui().setDepthImage(
					depthBuffer,
					depthWidth,
					depthHeight
				);
			}

			// Send ping-pong feedback image.
			// FIXME: This const-cast sucks.
			if (sendPingPongColorFeedback && !cameraFrame.colorImage.isEmpty())
				//sendFeedbackImage(const_cast<Image&>(cameraFrame.colorImage));
				sendImage(const_cast<Image&>(cameraFrame.colorImage));

			static unsigned long long count = 1; // FIXME: Use a real steady-rate timer.

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
};

//------------------------------------------------------------------------------


//------------------------------------------------------------------------------

struct UplinkServerDelegate : ServerDelegate
{
	void sendClearAllButtonsCommand()
	{
		_server->_currentSession->sendCustomCommand("button:clear:*");
	}

	void sendButtonCreateCommand(std::string buttonPngFilepath, std::string commandName)
	{
		CustomCommand customCommand;
		customCommand.command += "button:create:";
		customCommand.command += char(0);
		customCommand.command += commandName;
		customCommand.command += '\0';

		std::ifstream f(buttonPngFilepath, std::ios::binary);
		if (!f.is_open()) {
			std::cerr << "Cannot open:" << buttonPngFilepath << std::endl;
			std::cerr << "ERROR: wrong button file path" << std::endl;
			getchar();
		}
		std::string imageBytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
		customCommand.command.insert(customCommand.command.end(), imageBytes.begin(), imageBytes.end());

		_server->_currentSession->sendCustomCommand(customCommand);
	}

	void sendMessage( std::string msg)
	{
		CustomCommand customCommand;
		customCommand.command += "message:";
		customCommand.command += char(0);
		customCommand.command += msg;
		customCommand.command += '\0';

		_server->_currentSession->sendCustomCommand(customCommand);
	}

	void sendString(std::string msg)
	{
		CustomCommand customCommand;
		customCommand.command += msg;

		_server->_currentSession->sendCustomCommand(customCommand);
	}

	virtual ServerSession* newSession(int socketDescriptor, Server* server)
	{
		_server = server;

		return new UplinkSession(socketDescriptor, server);
	}

	virtual void onConnect(uintptr_t sessionId)
	{


		ScanectSessionSetup setup;
		SessionSettings settings; 
		setup.applyTo(settings); 




	//	std::cout << "Initializing sesstion with setup:\n" << settings.toString() << std::endl;
	//	_server->_currentSession->sendSessionSetup(SessionSetup());

	    _server->_currentSession->sendSessionSetup(setup);


		sendClearAllButtonsCommand();
		//For some reason relativ paths dont work when called from visual studio, even tho images are copied to the output folder
		sendButtonCreateCommand("record-button.png", "RecordButtonPressed");


		std::cin.get();
	}


	Server* _server;
};

//------------------------------------------------------------------------------

int main(int argc, char** argv)
{
	using namespace uplink;


	/*/
	RGBDStreamSessionSetup hack;

	MessageSerializer ser;
	uplink::Buffer buf;

	uplink::OutputStreamWriter writer(buf);
	ser.writeMessage(*writer.output, hack);


	uplink::InputStreamReader reader(buf);
	uplink::Message* msg = ser.readMessage(*reader.input);

	SessionSetup revSetup = msg->as<SessionSetup>();

	SessionSettings revSettings;
	revSetup.applyTo(revSettings);
	std::cout << "DecodedSettingsAs:\n" << revSettings.toString() << "\n\n MSG-kind Check: " << msg->kind() << "  should be: " << hack.kind() << std::endl;

	*/


	UplinkServerDelegate serverDelegate;

	// DesktopServer server("CaptureReceiverExample", UPLINK_SERVER_DEFAULT_TCP_PORT, &serverDelegate);
	DesktopServer server("UplinkCommServer", 6666, &serverDelegate);

	if (!server.startListening())
		return 1;

	server.ui().run();
}
