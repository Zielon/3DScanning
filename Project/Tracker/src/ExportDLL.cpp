#include "ExportDLL.h"

extern "C" __declspec(dllexport) void* createContext(const char* dataset_path){

	auto* tracker_context = new TrackerContext();

	tracker_context->m_videoStreamReader = new DatasetVideoStreamReader(dataset_path, false);

	tracker_context->m_videoStreamReader->startReading();
	//FIXME: Frame Info only set after first frame is read... FIXME: mb split this into seperate call?

	const auto height = tracker_context->m_videoStreamReader->m_height_depth;
	const auto width = tracker_context->m_videoStreamReader->m_width_depth;

	Matrix3f intrinsics = tracker_context->m_videoStreamReader->getCameraIntrinsics();
	const SystemParameters camera_parameters = SystemParameters(
		intrinsics(0, 0),
		intrinsics(1, 1),
		intrinsics(0, 2),
		intrinsics(1, 2),
		height,
		width,
		intrinsics
	);


	tracker_context->m_tracker = new Tracker(camera_parameters, ICPType::NON_LINEAR);
	tracker_context->m_fusion = new FusionGPU(camera_parameters);
	// Start consuming the point clouds buffer
	tracker_context->m_fusion->consume();

	return tracker_context;
}

extern "C" __declspec(dllexport) void * createSensorContext()
{
	TrackerContext* tracker_context = new TrackerContext();

	bool realtime = true, capture = false, verbose = false;

	#if _DEBUG
			capture = true;
			verbose = true;
	#endif


	//Sensor Class using OpenNI 2
	tracker_context->m_videoStreamReader = new Xtion2StreamReader(realtime, verbose, capture);

	tracker_context->m_videoStreamReader->startReading();
	//FIXME: Frame Info only set after first frame is read... FIXME: mb split this into seperate call?

	const auto height = tracker_context->m_videoStreamReader->m_height_depth;
	const auto width = tracker_context->m_videoStreamReader->m_width_depth;

	Matrix3f intrinsics = tracker_context->m_videoStreamReader->getCameraIntrinsics();
	const SystemParameters camera_parameters = SystemParameters(
		intrinsics(0, 0),
		intrinsics(1, 1),
		intrinsics(0, 2),
		intrinsics(1, 2),
		height,
		width,
		intrinsics
	);

	tracker_context->m_tracker = new Tracker(camera_parameters, NON_LINEAR);
	tracker_context->m_fusion = new Fusion(camera_parameters);
	// Start consuming the point clouds buffer
	tracker_context->m_fusion->consume();

	return tracker_context;
}

extern "C" __declspec(dllexport) int getImageWidth(void* context){
	auto* c = static_cast<TrackerContext*>(context);
	return c->m_videoStreamReader->m_width_rgb;
}

extern "C" __declspec(dllexport) int getImageHeight(void* context){
	auto* c = static_cast<TrackerContext*>(context);
	return c->m_videoStreamReader->m_height_rgb;
}

extern "C" __declspec(dllexport) void tracker(void* context, unsigned char* image, float* pose){

	auto* tracker_context = static_cast<TrackerContext*>(context);
	auto* tracker = tracker_context->m_tracker;

	cv::Mat rgb, depth;

	tracker_context->m_videoStreamReader->getNextFrame(rgb, depth, false);

	PointCloud* _target = new PointCloud(tracker->getCameraParameters(), depth, rgb);
	std::shared_ptr<PointCloud> current(_target);

	if (tracker_context->m_first_frame)
	{
		tracker->m_pose = Matrix4f::Identity(); 
		tracker_context->m_first_frame = false;
		tracker->m_previous_point_cloud = current;
		memcpy(pose, tracker->m_pose.data(), 16 * sizeof(float));
		return;
	}

	const Matrix4f delta = tracker->alignNewFrame(tracker->m_previous_point_cloud, current);

	tracker->m_pose *= delta;
	//tracker->m_pose = delta * tracker->m_pose;//Correct order (Juan opinion)
	current->m_pose_estimation = tracker->m_pose;


	if (tracker_context->enableReconstruction)
	{
		// Produce a new point cloud (add to the buffer)
		tracker_context->m_fusion->produce(std::shared_ptr<PointCloud>(tracker->m_previous_point_cloud));
	}

	tracker->m_previous_point_cloud = current;

	// Copy value to UNITY
	cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);
	std::memcpy(image, rgb.data, rgb.rows * rgb.cols * sizeof(unsigned char) * 3);
	memcpy(pose, tracker->m_pose.data(), 16 * sizeof(float));
}

extern "C" __declspec(dllexport) void getMeshInfo(void* context, __MeshInfo* mesh_info){
	auto* tracker_context = static_cast<TrackerContext*>(context);
	mesh_info->mesh = new Mesh();
	tracker_context->m_fusion->processMesh(*(mesh_info->mesh));
	mesh_info->m_index_count = mesh_info->mesh->m_triangles.size() * 3;
	mesh_info->m_vertex_count = mesh_info->mesh->m_vertices.size();
}

extern "C" __declspec(dllexport) void getMeshBuffers(__MeshInfo* mesh_info, float* pVB, int* pIB){
	memcpy(pVB, mesh_info->mesh->m_vertices.data(), mesh_info->m_vertex_count * 3 * sizeof(float));
	memcpy(pIB, mesh_info->mesh->m_triangles.data(), mesh_info->m_index_count * sizeof(int));
	delete mesh_info->mesh;
}

extern "C" __declspec(dllexport) void enableReconstruction(void* context, bool enable)
{
	auto* tracker_context = static_cast<TrackerContext*>(context);
	tracker_context->enableReconstruction = enable;
}
