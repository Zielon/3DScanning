#include <iostream>
//OpenNI
#include <XnOpenNI.h>
#include <XnLog.h>
#include <XnCppWrapper.h>
#include <XnFPSCalculator.h>

#define SAMPLE_XML_PATH "SamplesConfig.xml"

using namespace std;
using namespace xn;

//---------------------------------------------------------------------------
// Macros
//---------------------------------------------------------------------------
#define CHECK_RC(rc, what)											\
	if (rc != XN_STATUS_OK)											\
	{																\
		printf("%s failed: %s\n", what, xnGetStatusString(rc));		\
		return rc;													\
	}


XnBool fileExists(const char *fn)
{
	XnBool exists;
	xnOSDoesFileExist(fn, &exists);
	return exists;
}

int main()
{
	XnStatus nRetVal = XN_STATUS_OK;

	Context context;
	ScriptNode scriptNode;
	EnumerationErrors errors;

	const char *fn = NULL;
	
	//Check if the configuration path exists
	if (fileExists(SAMPLE_XML_PATH)) {
		fn = SAMPLE_XML_PATH;
	}else {
		printf("Could not find '%s' nor '%s'. Aborting.\n", SAMPLE_XML_PATH, SAMPLE_XML_PATH);
		return XN_STATUS_ERROR;
	}

	printf("Reading config from: '%s'\n", fn);

	//Create context from configuration file
	nRetVal = context.InitFromXmlFile(fn, scriptNode, &errors);

	if (nRetVal == XN_STATUS_NO_NODE_PRESENT)
	{
		XnChar strError[1024];
		errors.ToString(strError, 1024);
		printf("%s\n", strError);
		return (nRetVal);
	}
	else if (nRetVal != XN_STATUS_OK)
	{
		printf("Open failed: %s\n", xnGetStatusString(nRetVal));
		return (nRetVal);
	}

	//Setting image generator (RGB color)
	ImageGenerator color_generator;
	nRetVal = context.FindExistingNode(XN_NODE_TYPE_IMAGE, color_generator);
	CHECK_RC(nRetVal, "Find color generator");

	//Setting depth degenerator
	DepthGenerator depth_generator;
	nRetVal = context.FindExistingNode(XN_NODE_TYPE_DEPTH, depth_generator);
	CHECK_RC(nRetVal, "Find depth generator");

	ImageMetaData colorMD;
	DepthMetaData depthMD;

	color_generator.GetMetaData(colorMD);
	depth_generator.GetMetaData(depthMD);

	//Color image must be RGBformat.
	if (colorMD.PixelFormat() != XN_PIXEL_FORMAT_RGB24)
	{
		printf("The device image format must be RGB24\n");
		return 1;
	}

	// Color resolution must be equal to depth resolution
	if (colorMD.FullXRes() != depthMD.FullXRes() || colorMD.FullYRes() != depthMD.FullYRes())
	{
		printf("The device depth and image resolution must be equal!\n");
		return 1;
	}

	XnFPSData xnFPS;
	nRetVal = xnFPSInit(&xnFPS, 180);
	CHECK_RC(nRetVal, "FPS Init");

	//Processing each frame until the user stops the process by hitting a key
	while (!xnOSWasKeyboardHit())
	{
		/*nRetVal = context.WaitOneUpdateAll(depth_generator);

		if (nRetVal != XN_STATUS_OK)
		{
			printf("UpdateData failed: %s\n", xnGetStatusString(nRetVal));
			continue;
		}*/

		//Read a new frame
		nRetVal = context.WaitAnyUpdateAll();

		if (nRetVal != XN_STATUS_OK)
		{
			printf("ReadData failed: %s\n", xnGetStatusString(nRetVal));
			continue;
		}

		xnFPSMarkFrame(&xnFPS);

		//Getting data from generator
		color_generator.GetMetaData(colorMD);
		depth_generator.GetMetaData(depthMD);

		const XnUInt8* color_map = colorMD.Data();
		const XnDepthPixel* depth_map = depthMD.Data();

		printf("Color frame %d: resolution (%d, %d), bytes %d\n", colorMD.FrameID(), colorMD.XRes(), colorMD.YRes(), colorMD.DataSize());
		printf("Depth frame %d: resolution (%d, %d), bytes %d\n", depthMD.FrameID(), depthMD.XRes(), depthMD.YRes(), depthMD.DataSize());

		//cin.get();
		//printf("Color frame %d: (center: %u, FPS: %f)\n", colorMD.FrameID(), colorMD(colorMD.XRes() / 2, colorMD.YRes() / 2), xnFPSCalc(&xnFPS));
		//printf("Depth frame %d: (center: %u, FPS: %f)\n", depthMD.FrameID(), depthMD(depthMD.XRes() / 2, depthMD.YRes() / 2), xnFPSCalc(&xnFPS));
	}

	//Release resources
	color_generator.Release();
	depth_generator.Release();
	scriptNode.Release();
	context.Release();


	return 0;
}