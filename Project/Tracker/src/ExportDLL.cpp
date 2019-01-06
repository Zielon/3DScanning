#include "ExportDLL.h"

#ifdef _WIN32

#define PIXEL_STEPS 8

extern "C" __declspec(dllexport) void * createContext(char* dataset_path) {

	TrackerContext* c = new  TrackerContext();
	c->videoStreamReader = new DatasetVideoStreamReader(dataset_path, true);
	c->videoStreamReader->startReading(); //FIXME: Frame Info only set after first frame is read... FIXME: mb split this into seperate call?

	Matrix3f intrinsics = c->videoStreamReader->getCameraIntrinsics(); 

    c->tracker = new Tracker(intrinsics(0,0), intrinsics(1,1), intrinsics(0,2), intrinsics(1,2),
		c->videoStreamReader->m_width_depth, c->videoStreamReader->m_height_depth);

	c->fusion = new Fusion(c->videoStreamReader->m_width_depth, c->videoStreamReader->m_height_depth, PIXEL_STEPS);

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
	c->tracker->backprojectFrame(depth, newFrameVerts, PIXEL_STEPS);

	c->fusion->m_currentFrameIndexBuffer.clear(); 
	c->fusion->generateMeshFromVertices(newFrameVerts, c->fusion->m_currentFrameIndexBuffer);

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



extern "C" __declspec(dllexport) int getVertexCount(void* context)
{
	TrackerContext * c = static_cast<TrackerContext*>(context);

	return c->tracker->m_previousFrameVerts.size(); 
}

extern "C" __declspec(dllexport) void getVertexBuffer(void* context, float *vertices)
{
	TrackerContext * c = static_cast<TrackerContext*>(context);

	memcpy(vertices, c->tracker->m_previousFrameVerts.data(), c->tracker->m_previousFrameVerts.size()* sizeof(Vector3f));
}

extern "C" __declspec(dllexport) int getIndexCount(void* context)
{
	TrackerContext * c = static_cast<TrackerContext*>(context);

	return c->fusion->m_currentFrameIndexBuffer.size(); 
}

extern "C" __declspec(dllexport) void getIndexBuffer(void* context, int* indices)
{
	TrackerContext * c = static_cast<TrackerContext*>(context);

	memcpy(indices, c->fusion->m_currentFrameIndexBuffer.data(), c->fusion->m_currentFrameIndexBuffer.size() * sizeof(int));
}

#endif