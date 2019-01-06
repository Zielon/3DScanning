#include "ExportDLL.h"

#ifdef _WIN32

extern "C" __declspec(dllexport) void * createContext(char* dataset_path) {

	TrackerContext* c = new  TrackerContext();
	c->videoStreamReader = new DatasetVideoStreamReader(dataset_path, true);
	c->videoStreamReader->startReading(); //FIXME: Frame Info only set after first frame is read... FIXME: mb split this into seperate call?

	Matrix3f intrinsics = c->videoStreamReader->getCameraIntrinsics(); 

    c->tracker = new Tracker(intrinsics(0,0), intrinsics(1,1), intrinsics(0,2), intrinsics(1,2),
		c->videoStreamReader->m_width_depth, c->videoStreamReader->m_height_depth);

    return c;
}

extern "C" __declspec(dllexport) void trackerCameraPose(void *context, unsigned char *image, float *pose, int w, int h) {

    Tracker *tracker = static_cast<TrackerContext*>(context)->tracker;

    //tracker->computerCameraPose(image, pose, w, h);
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

extern "C" __declspec(dllexport) void dllMain(void *context, unsigned char *image, float *pose)
{
	TrackerContext * c = static_cast<TrackerContext*>(context);

    cv::Mat rgb, depth;

	bool firstFrame = c->tracker->m_previousFrameVerts.size() == 0;

    c->videoStreamReader->getNextFrame(rgb, depth, false);

	std::vector<Vector3f> newFrameVerts;

    //DEBUG
    /*cv::imshow("dllMain", rgb);
    cv::waitKey(1);*/
	c->tracker->backprojectFrame(depth, newFrameVerts, 8);

	if (firstFrame) // first frame
	{
		Matrix4f id = Matrix4f::Identity(); 
		memcpy(pose, id.data(), 16 * sizeof(float)); 
	}
	else
	{
		c->tracker->alignNewFrame(newFrameVerts, c->tracker->m_previousFrameVerts, pose);
	}

	//TODO: real time mesh generation here

	c->tracker->m_previousFrameVerts = newFrameVerts;

    /*DEBUG*
    cv::imshow("dllMain", rgb);
    cv::waitKey(1);
    /**/

    //So turns out opencv actually uses bgr not rgb...
    //no more opencv computations after this point
    cv::cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);
    std::memcpy(image, rgb.data, rgb.rows * rgb.cols * sizeof(unsigned char) * 3);
}

#endif