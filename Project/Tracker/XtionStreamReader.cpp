#include "XtionStreamReader.h"

XtionStreamReader::XtionStreamReader(bool realtime) {

	/*XnStatus nRetVal = XN_STATUS_OK;

	Context context;
	ScriptNode scriptNode;
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

	printf("XN context return value: %d\n", nRetVal);*/

	this->m_realtime = realtime;
}

XtionStreamReader::~XtionStreamReader(){
	//Release resources
	//scriptNode.Release();
	//this->context.Release();
}

XnBool XtionStreamReader::fileExists(const char *fn)
{
	XnBool exists;
	//xnOSDoesFileExist(fn, &exists);
	return exists;
}

bool XtionStreamReader::nextFrameAvailable() {

	return false;
}

int XtionStreamReader::getSequentialFrame(cv::Mat &rgb, cv::Mat &depth) {

	return 0;
}

int XtionStreamReader::getLatestFrame(cv::Mat &rgb, cv::Mat &depth) {

	return 0;
}

bool XtionStreamReader::startReading() {

	return true;
}

bool XtionStreamReader::stopReading() {

	return true;
}