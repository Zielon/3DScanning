#include "ExportDLL.h"
#include "data-stream/headers/DatasetVideoStreamReader.h"
#include <opencv2/imgproc/imgproc.hpp>

#ifdef _WIN32

__declspec(dllexport) void * createContext() {

    Context* c = new Context();
	c->videoStreamReader = new DatasetVideoStreamReader(DATASET_DIR, true);
	c->videoStreamReader->startReading(); //FIXME: Frame Info only set after first frame is read... FIXME: mb split this into seperate call?


	Matrix3f intrinsics = c->videoStreamReader->getCameraIntrinsics(); 

    c->tracker = new Tracker(intrinsics(0,0), intrinsics(1,1), intrinsics(0,2), intrinsics(1,2),
		c->videoStreamReader->m_width_depth, c->videoStreamReader->m_height_depth);


    return c;
}

__declspec(dllexport) void trackerCameraPose(void *context, byte *image, float *pose, int w, int h) {

    Tracker *tracker = static_cast<Context*>(context)->tracker;

 //   tracker->computerCameraPose(image, pose, w, h);
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

	bool firstFrame = c->tracker->m_previousFrame.size() == 0;

    c->videoStreamReader->getNextFrame(rgb, depth, true);

	std::vector<Vector3f> newFrameVerts;

	c->tracker->backprojectFrame(depth, newFrameVerts, 8);

	if (firstFrame) // first frame
	{
		Matrix4f id = Matrix4f::Identity(); 
		memcpy(pose, id.data(), 16 * sizeof(float)); 
	}
	else
	{
		c->tracker->alignNewFrame(newFrameVerts, c->tracker->m_previousFrame, pose);
	}

	//TODO: mesh generation here

	c->tracker->m_previousFrame = newFrameVerts;

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