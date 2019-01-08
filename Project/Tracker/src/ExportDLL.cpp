#include "ExportDLL.h"

#ifdef _WIN32

extern "C" __declspec(dllexport) void* createContext(char* dataset_path){

	TrackerContext* tracker_context = new TrackerContext();

	#if _DEBUG
	tracker_context->videoStreamReader = new DatasetVideoStreamReader(dataset_path, false);
	#else
	tracker_context->videoStreamReader = new DatasetVideoStreamReader(dataset_path, true);
	#endif

	tracker_context->videoStreamReader->startReading();
	//FIXME: Frame Info only set after first frame is read... FIXME: mb split this into seperate call?

	Matrix3f intrinsics = tracker_context->videoStreamReader->getCameraIntrinsics();

	const CameraParameters camera_parameters = CameraParameters(
		intrinsics(0, 0),
		intrinsics(1, 1),
		intrinsics(0, 2),
		intrinsics(1, 2),
		tracker_context->videoStreamReader->m_height_depth,
		tracker_context->videoStreamReader->m_width_depth
	);

	tracker_context->tracker = new Tracker(camera_parameters);

	return tracker_context;
}

extern "C" __declspec(dllexport) void trackerCameraPose(void* context, unsigned char* image, float* pose, int w, int h){

	Tracker* tracker = static_cast<TrackerContext*>(context)->tracker;

	//tracker->computerCameraPose(image, pose, w, h);
}

extern "C" __declspec(dllexport) int getImageWidth(void* context){
	TrackerContext* c = static_cast<TrackerContext*>(context);
	return c->videoStreamReader->m_width_rgb;
}

extern "C" __declspec(dllexport) int getImageHeight(void* context){
	TrackerContext* c = static_cast<TrackerContext*>(context);
	return c->videoStreamReader->m_height_rgb;
}

extern "C" __declspec(dllexport) void dllMain(void* context, unsigned char* image, float* pose){
	TrackerContext* tracker_context = static_cast<TrackerContext*>(context);

	cv::Mat rgb, depth;

	bool is_first_frame = tracker_context->tracker->m_previous_point_cloud.getPoints().size() == 0;

	tracker_context->videoStreamReader->getNextFrame(rgb, depth, false);

	const PointCloud source = PointCloud(tracker_context->tracker->getCameraParameters(), depth, 32);

	if (is_first_frame) // first frame
	{
		Matrix4f id = Matrix4f::Identity();
		memcpy(pose, id.data(), 16 * sizeof(float));
	}
	else
	{
		tracker_context->tracker->alignNewFrame(source, tracker_context->tracker->m_previous_point_cloud, pose);
	}

	//TODO: real time mesh generation here

	tracker_context->tracker->m_previous_point_cloud = source;

	/*DEBUG*
	cv::imshow("dllMain", rgb);
	cv::waitKey(1);
	/**/

	//So turns out opencv actually uses bgr not rgb...
	//no more opencv computations after this point
	cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);
	std::memcpy(image, rgb.data, rgb.rows * rgb.cols * sizeof(unsigned char) * 3);
}

#endif
