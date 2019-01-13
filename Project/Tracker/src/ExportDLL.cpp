#include "ExportDLL.h"

extern "C" __declspec(dllexport) void* createContext(char* dataset_path){

	TrackerContext* tracker_context = new TrackerContext();

	#if _DEBUG
	tracker_context->m_videoStreamReader = new DatasetVideoStreamReader(dataset_path, false);
	#else
	tracker_context->m_videoStreamReader = new DatasetVideoStreamReader(dataset_path, true);
	#endif

	tracker_context->m_videoStreamReader->startReading();
	//FIXME: Frame Info only set after first frame is read... FIXME: mb split this into seperate call?

	const auto height = tracker_context->m_videoStreamReader->m_height_depth;
	const auto width = tracker_context->m_videoStreamReader->m_width_depth;

	Matrix3f intrinsics = tracker_context->m_videoStreamReader->getCameraIntrinsics();
	const CameraParameters camera_parameters = CameraParameters(
		intrinsics(0, 0),
		intrinsics(1, 1),
		intrinsics(0, 2),
		intrinsics(1, 2),
		height,
		width
	);

	tracker_context->m_tracker = new Tracker(camera_parameters);
	tracker_context->m_fusion = new Fusion(camera_parameters);
	// Start consuming the point clouds buffer
	tracker_context->m_fusion->consume();

	return tracker_context;
}

void * createSensorContext(char *sensor_path, bool useOpenni2)
{
	TrackerContext* tracker_context = new TrackerContext();

	bool realtime = true, capture = false, verbose = false;

	#if _DEBUG
		capture = true;
		verbose = true;
	#endif

	//Sensor Class using OpenNI 2
	if (useOpenni2) {
		tracker_context->m_videoStreamReader = new Xtion2StreamReader(realtime, verbose, capture);
	}
	else {
		tracker_context->m_videoStreamReader = new XtionStreamReader(sensor_path, realtime, verbose, capture);
	}

	tracker_context->m_videoStreamReader->startReading();
	//FIXME: Frame Info only set after first frame is read... FIXME: mb split this into seperate call?

	Matrix3f intrinsics = tracker_context->m_videoStreamReader->getCameraIntrinsics();
	const CameraParameters camera_parameters = CameraParameters(
		intrinsics(0, 0),
		intrinsics(1, 1),
		intrinsics(0, 2),
		intrinsics(1, 2),
		tracker_context->m_videoStreamReader->m_height_depth,
		tracker_context->m_videoStreamReader->m_width_depth
	);

	tracker_context->m_tracker = new Tracker(camera_parameters);

	return tracker_context;
}

int getNextFrame(void * context, unsigned char * image)
{
	TrackerContext* tracker_context = static_cast<TrackerContext*>(context);

	cv::Mat rgb, depth;

	tracker_context->m_videoStreamReader->getNextFrame(rgb, depth, false);

	//Copy color frame
	std::memcpy(image, rgb.data, rgb.rows * rgb.cols * sizeof(unsigned char) * 3);

	//Test
	/*cv::cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);//OpenCV uses bgr not rgb format
	cv::imshow("dllTest", rgb);
	cv::waitKey(1);*/

	return tracker_context->m_videoStreamReader->m_Status;
}

extern "C" __declspec(dllexport) void trackerCameraPose(void* context, unsigned char* image, float* pose, int w, int h){

	Tracker* tracker = static_cast<TrackerContext*>(context)->m_tracker;

	//m_tracker->computerCameraPose(image, pose, w, h);
}

extern "C" __declspec(dllexport) int getImageWidth(void* context){
	TrackerContext* c = static_cast<TrackerContext*>(context);
	return c->m_videoStreamReader->m_width_rgb;
}

extern "C" __declspec(dllexport) int getImageHeight(void* context){
	TrackerContext* c = static_cast<TrackerContext*>(context);
	return c->m_videoStreamReader->m_height_rgb;
}

extern "C" __declspec(dllexport) void dllMain(void* context, unsigned char* image, float* pose, bool use_fusion){
	
	TrackerContext* tracker_context = static_cast<TrackerContext*>(context);

	cv::Mat rgb, depth;

	const bool is_first_frame = tracker_context->m_tracker->m_previous_point_cloud == nullptr;

	tracker_context->m_videoStreamReader->getNextFrame(rgb, depth, false);

	//Test
	/*cv::cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);//OpenCV uses bgr not rgb format
	cv::imshow("dllTest", rgb);
	cv::waitKey(1);*/

	PointCloud* source = new PointCloud(tracker_context->m_tracker->getCameraParameters(), depth);

	if (is_first_frame) // first frame
	{
		Matrix4f id = Matrix4f::Identity();
		memcpy(pose, id.data(), 16 * sizeof(float));
		tracker_context->m_tracker->m_previous_point_cloud = source;
		return;
	}
	else
	{
		tracker_context->m_tracker->alignNewFrame(source, tracker_context->m_tracker->m_previous_point_cloud, pose);
	}

	// Produce a new point cloud (add to the buffer)
	
	if (use_fusion) {
		tracker_context->m_fusion->produce(tracker_context->m_tracker->m_previous_point_cloud);//Crash unity
	}

	// Safe the last frame reference
	tracker_context->m_tracker->m_previous_point_cloud = source;

	//Copy color frame
	std::memcpy(image, rgb.data, rgb.rows * rgb.cols * sizeof(unsigned char) * 3);
}

extern "C" __declspec(dllexport) int getVertexCount(void* context){
	TrackerContext* c = static_cast<TrackerContext*>(context);
	return c->m_tracker->m_previous_point_cloud->getPoints().size();
}

extern "C" __declspec(dllexport) void getVertexBuffer(void* context, float* vertices){
	TrackerContext* c = static_cast<TrackerContext*>(context);
	memcpy(vertices, c->m_tracker->m_previous_point_cloud->getPoints().data(),
	       c->m_tracker->m_previous_point_cloud->getPoints().size() * sizeof(Vector3f));
}

extern "C" __declspec(dllexport) int getIndexCount(void* context){
	TrackerContext* c = static_cast<TrackerContext*>(context);
	return c->m_fusion->m_currentIndexBuffer.size();
}

extern "C" __declspec(dllexport) void getIndexBuffer(void* context, int* indices){
	TrackerContext* c = static_cast<TrackerContext*>(context);
	memcpy(indices, c->m_fusion->m_currentIndexBuffer.data(), c->m_fusion->m_currentIndexBuffer.size() * sizeof(int));
}

void getNormalBuffer(void* context, float* normals){
	TrackerContext* c = static_cast<TrackerContext*>(context);
	memcpy(normals, c->m_tracker->m_previous_point_cloud->getNormals().data(),
	       c->m_tracker->m_previous_point_cloud->getNormals().size() * sizeof(Vector3f));
}
