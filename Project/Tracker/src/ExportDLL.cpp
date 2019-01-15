#include "ExportDLL.h"

extern "C" __declspec(dllexport) void* createContext(const char* dataset_path){

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
		width,
		intrinsics
	);

	tracker_context->m_tracker = new Tracker(camera_parameters);
	tracker_context->m_fusion = new Fusion(camera_parameters);
	// Start consuming the point clouds buffer
	tracker_context->m_fusion->consume();

	return tracker_context;
}

extern "C" __declspec(dllexport) void trackerCameraPose(void* context, unsigned char* image, float* pose){

	TrackerContext* tracker_context = static_cast<TrackerContext*>(context);

	cv::Mat rgb, depth;

	const bool is_first_frame = tracker_context->m_tracker->m_previous_point_cloud == nullptr;

	tracker_context->m_videoStreamReader->getNextFrame(rgb, depth, false);

	PointCloud* source = new PointCloud(tracker_context->m_tracker->getCameraParameters(), depth, rgb);

	//source->transform(tracker_context->m_tracker->m_previous_pose);

	if (is_first_frame) // first frame
	{
		tracker_context->m_tracker->m_previous_pose = Matrix4f::Identity();
		tracker_context->m_tracker->m_previous_point_cloud = source;

		memcpy(pose, tracker_context->m_tracker->m_previous_pose.data(), 16 * sizeof(float));

		return;
	}
	else
	{
		//std::cout << "Test Tracker Camera Pose" << std::endl;

		std::cout << "Previous Pose" << std::endl;
		std::cout << tracker_context->m_tracker->m_previous_pose << std::endl;

		Matrix4f deltaPose  = tracker_context->m_tracker->alignNewFrame(source, tracker_context->m_tracker->m_previous_point_cloud);

		//tracker_context->m_tracker->m_previous_pose = deltaPose * tracker_context->m_tracker->m_previous_pose;
		tracker_context->m_tracker->m_previous_pose =  tracker_context->m_tracker->m_previous_pose * deltaPose;

		std::cout << "Delta Pose" << std::endl;
		std::cout << deltaPose << std::endl;

		memcpy(pose, tracker_context->m_tracker->m_previous_pose.data(), 16 * sizeof(float));
	}

	// Safe the last frame reference
	tracker_context->m_tracker->m_previous_point_cloud = source;

	//So turns out opencv actually uses bgr not rgb...
	//no more opencv computations after this point
	cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);
	std::memcpy(image, rgb.data, rgb.rows * rgb.cols * sizeof(unsigned char) * 3);
}

extern "C" __declspec(dllexport) int getImageWidth(void* context){
	TrackerContext* c = static_cast<TrackerContext*>(context);
	return c->m_videoStreamReader->m_width_rgb;
}

extern "C" __declspec(dllexport) int getImageHeight(void* context){
	TrackerContext* c = static_cast<TrackerContext*>(context);
	return c->m_videoStreamReader->m_height_rgb;
}

extern "C" __declspec(dllexport) void dllMain(void* context, unsigned char* image, float* pose){
	
	TrackerContext* tracker_context = static_cast<TrackerContext*>(context);

	cv::Mat rgb, depth;

	const bool is_first_frame = tracker_context->m_tracker->m_previous_point_cloud == nullptr;

	tracker_context->m_videoStreamReader->getNextFrame(rgb, depth, false);

	PointCloud* source = new PointCloud(tracker_context->m_tracker->getCameraParameters(), depth, rgb);

	if (is_first_frame) // first frame
	{
		tracker_context->m_tracker->m_previous_pose = Matrix4f::Identity();
		tracker_context->m_tracker->m_previous_point_cloud = source;

		memcpy(pose, tracker_context->m_tracker->m_previous_pose.data(), 16 * sizeof(float));
		
		return;
	}
	else
	{
		Matrix4f deltaPose = tracker_context->m_tracker->alignNewFrame(source, tracker_context->m_tracker->m_previous_point_cloud);

		tracker_context->m_tracker->m_previous_pose = deltaPose * tracker_context->m_tracker->m_previous_pose;
	}

	// Produce a new point cloud (add to the buffer)
	tracker_context->m_fusion->produce(tracker_context->m_tracker->m_previous_point_cloud);

	// Safe the last frame reference
	tracker_context->m_tracker->m_previous_point_cloud = source;

	//So turns out opencv actually uses bgr not rgb...
	//no more opencv computations after this point
	cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);
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
