#include "ExportDLL.h"

/*extern "C" __declspec(dllexport) int test() {

	return 8;
}*/

#ifdef _WIN32

extern "C" __declspec(dllexport) void * createContext() {

	TrackerContext* c = new  TrackerContext();
    c->tracker = new Tracker();

	#ifdef DATASET

    c->videoStreamReader = new DatasetVideoStreamReader(DATASET_DIR, true);

	#endif

	#ifdef XTION_SENSOR

		c->videoStreamReader = new XtionStreamReader(true);

	#endif

    c->videoStreamReader->startReading(); //FIXME: mb split this into seperate call?

    return c;
}

extern "C" __declspec(dllexport) void trackerCameraPose(void *context, byte *image, float *pose, int w, int h) {

    Tracker *tracker = static_cast<TrackerContext*>(context)->tracker;

    tracker->computerCameraPose(image, pose, w, h);
}

extern "C" __declspec(dllexport) int getImageWidth(void *context)
{
	TrackerContext * c = static_cast<TrackerContext*>(context);
    return c->videoStreamReader->m_width_rgb;
}

extern "C" __declspec(dllexport) int getImageHeight(void *context)
{
	TrackerContext * c = static_cast<TrackerContext*>(context);
    return c->videoStreamReader->m_height_rgb;
}

extern "C" __declspec(dllexport) void dllMain(void *context, byte *image, float *pose)
{
	TrackerContext * c = static_cast<TrackerContext*>(context);

    cv::Mat rgb, depth;

    c->videoStreamReader->getNextFrame(rgb, depth, true);

    c->tracker->alignToNewFrame(rgb, depth, pose);

    //DEBUG
    /*cv::imshow("dllMain", rgb);
    cv::waitKey(1);*/

    //So turns out opencv actually uses bgr not rgb...
    //no more opencv computations after this point
    cv::cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);
    std::memcpy(image, rgb.data, rgb.rows * rgb.cols * sizeof(byte) * 3);
}

#endif