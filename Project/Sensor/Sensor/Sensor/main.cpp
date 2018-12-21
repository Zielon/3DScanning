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

	//Setting depth degenerator
	DepthGenerator depth;
	nRetVal = context.FindExistingNode(XN_NODE_TYPE_DEPTH, depth);
	CHECK_RC(nRetVal, "Find depth generator");

	XnFPSData xnFPS;
	nRetVal = xnFPSInit(&xnFPS, 180);
	CHECK_RC(nRetVal, "FPS Init");

	cin.get();

	//Processing each frame
	DepthMetaData depthMD;

	while (!xnOSWasKeyboardHit())
	{
		nRetVal = context.WaitOneUpdateAll(depth);
		if (nRetVal != XN_STATUS_OK)
		{
			printf("UpdateData failed: %s\n", xnGetStatusString(nRetVal));
			continue;
		}

		xnFPSMarkFrame(&xnFPS);

		//Getting depth data from depth generator
		depth.GetMetaData(depthMD);
		const XnDepthPixel* pDepthMap = depthMD.Data();

		printf("Frame %d Middle point is: %u. FPS: %f\n", depthMD.FrameID(), depthMD(depthMD.XRes() / 2, depthMD.YRes() / 2), xnFPSCalc(&xnFPS));
	}

	//Release resources
	depth.Release();
	scriptNode.Release();
	context.Release();


	return 0;
}