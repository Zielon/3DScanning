#include "ExportDLL.h"

#include "DatasetVideoStreamReader.h"
#include <opencv2/imgproc/imgproc.hpp>

#ifdef _WIN32

__declspec(dllexport) void * createContext() {

	Context* c = new Context(); 
	c->tracker = new Tracker(); 
	c->videoStreamReader = new DatasetVideoStreamReader(DATASET_DIR, true); 

	c->videoStreamReader->startReading(); //FIXME: mb split this into seperate call?


	return c; 

}

__declspec(dllexport) void trackerCameraPose(void *context, byte *image, float *pose, int w, int h) {

	Tracker *tracker = static_cast<Context*>(context)->tracker;

	tracker->computerCameraPose(image, pose, w, h);
}

extern "C" __declspec(dllexport) int getImageWidth(void *context)
{
	Context * c = static_cast<Context*>(context);
	return c->videoStreamReader->m_width_rgb;

}
extern "C" __declspec(dllexport) int getImageHeight(void *context)
{
	Context * c = static_cast<Context*>(context);
	return c->videoStreamReader->m_height_rgb;

}

__declspec(dllexport) void dllMain(void *context, byte *image, float *pose)
{
	Context * c = static_cast<Context*>(context); 

	cv::Mat rgb, depth; 

	c->videoStreamReader->getNextFrame(rgb, depth, true);

	c->tracker->alignToNewFrame(rgb, depth, pose); 
	
	/*DEBUG*
	cv::imshow("dllMain", rgb); 
	cv::waitKey(1); 
	/**/

	//So turns out opencv actually uses bgr not rgb...
	//no more opencv computations after this point
	cv::cvtColor(rgb, rgb, cv::COLOR_BGR2RGB); 
	std::memcpy(image, rgb.data, rgb.rows * rgb.cols * sizeof(byte) * 3);

}

#endif