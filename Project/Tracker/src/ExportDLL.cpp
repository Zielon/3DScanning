#include "ExportDLL.h"

extern "C" __declspec(dllexport) void* createContext(const char* dataset_path){

	auto* tracker_context = new TrackerContext();

	tracker_context->m_videoStreamReader = new DatasetVideoStreamReader(dataset_path, false);
	tracker_context->m_videoStreamReader->startReading();

	const auto height = tracker_context->m_videoStreamReader->m_height_depth;
	const auto width = tracker_context->m_videoStreamReader->m_width_depth;

	Matrix3f intrinsics = tracker_context->m_videoStreamReader->getCameraIntrinsics();
	const SystemParameters system_parameters = SystemParameters(
		intrinsics(0, 0),
		intrinsics(1, 1),
		intrinsics(0, 2),
		intrinsics(1, 2),
		height,
		width,
		intrinsics
	);

	tracker_context->m_tracker = new Tracker(system_parameters, CUDA);
	tracker_context->m_fusion = new FusionGPU(system_parameters);
	tracker_context->m_fusion->consume();

	return tracker_context;
}

extern "C" __declspec(dllexport) void* createSensorContext(const char* dataset_path){
	TrackerContext* tracker_context = new TrackerContext();

	bool realtime = true, capture = false, verbose = false;

	#if _DEBUG
			capture = true;
			verbose = true;
	#endif

	//Sensor Class using OpenNI 2
	tracker_context->m_videoStreamReader = new Xtion2StreamReader(realtime, verbose, capture);
	tracker_context->m_videoStreamReader->startReading();

	const auto height = tracker_context->m_videoStreamReader->m_height_depth;
	const auto width = tracker_context->m_videoStreamReader->m_width_depth;

	Matrix3f intrinsics = tracker_context->m_videoStreamReader->getCameraIntrinsics();
	const SystemParameters system_parameters = SystemParameters(
		intrinsics(0, 0),
		intrinsics(1, 1),
		intrinsics(0, 2),
		intrinsics(1, 2),
		height,
		width,
		intrinsics
	);
	tracker_context->m_tracker = new Tracker(system_parameters, CUDA);
	tracker_context->m_fusion = new FusionGPU(system_parameters);

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

	PointCloud* _target = new PointCloud(tracker->getSystemParameters(), depth, rgb);
	std::shared_ptr<PointCloud> current(_target);

	if (tracker_context->m_first_frame)
	{
		tracker->m_pose = Matrix4f::Identity();
		tracker_context->m_first_frame = false;
		tracker->m_previous_point_cloud = current;
		memcpy(pose, tracker->m_pose.data(), 16 * sizeof(float));
		return;
	}

	const Matrix4f new_pose = tracker->m_icp->estimatePose(tracker->m_previous_point_cloud, current);

	tracker->m_pose = new_pose;
	current->m_pose_estimation = new_pose;

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

extern "C" __declspec(dllexport) void enableReconstruction(void* context, bool enable){
	auto* tracker_context = static_cast<TrackerContext*>(context);
	tracker_context->enableReconstruction = enable;
}


extern "C" __declspec(dllexport) void getFrame(void* context, unsigned char* image, bool record)
{
	auto* tracker_context = static_cast<TrackerContext*>(context);

	cv::Mat rgb, depth;

	tracker_context->m_videoStreamReader->getNextFrame(rgb, depth, false);

	tracker_context->rgb_recording.push_back(rgb);
	tracker_context->depth_recording.push_back(depth); 


	// Copy value to UNITY
	cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);
	std::memcpy(image, rgb.data, rgb.rows * rgb.cols * sizeof(unsigned char) * 3);

}


extern "C" __declspec(dllexport) void computeOfflineReconstruction(void* context, __MeshInfo* mesh_info, float* pose)
{
	TrackerContext* tracker_context = static_cast<TrackerContext*>(context);
	const auto height = tracker_context->m_videoStreamReader->m_height_depth;
	const auto width = tracker_context->m_videoStreamReader->m_width_depth;

	Matrix3f intrinsics = tracker_context->m_videoStreamReader->getCameraIntrinsics();
	const SystemParameters system_parameters = SystemParameters(
		intrinsics(0, 0),
		intrinsics(1, 1),
		intrinsics(0, 2),
		intrinsics(1, 2),
		height,
		width,
		intrinsics
	);

	Tracker* tracker = tracker_context->m_tracker; 

	tracker_context->m_first_frame = true;

	auto rgbIt = tracker_context->rgb_recording.begin(); 

	for (auto depth : tracker_context->depth_recording)
	{
		auto rgb = *rgbIt++; 

		PointCloud* _target = new PointCloud(tracker->getSystemParameters(), depth, rgb);
		std::shared_ptr<PointCloud> current(_target);

		if (tracker_context->m_first_frame)
		{
			tracker->m_pose = Matrix4f::Identity();
			tracker_context->m_first_frame = false;
			tracker->m_previous_point_cloud = current;
			continue;
		}

		const Matrix4f delta = tracker->alignNewFrame(tracker->m_previous_point_cloud, current);

		tracker->m_pose *= delta;
		//tracker->m_pose = delta * tracker->m_pose;//Correct order (Juan opinion)
		current->m_pose_estimation = tracker->m_pose;

		// Produce a new point cloud (add to the buffer)
		tracker_context->m_fusion->produce(std::shared_ptr<PointCloud>(tracker->m_previous_point_cloud));
		tracker->m_previous_point_cloud = current;

	}
	tracker_context->rgb_recording.clear();
	tracker_context->depth_recording.clear();

	tracker_context->m_fusion->wait(); 

	mesh_info->mesh = new Mesh();
	tracker_context->m_fusion->processMesh(*(mesh_info->mesh));
	mesh_info->m_index_count = mesh_info->mesh->m_triangles.size() * 3;
	mesh_info->m_vertex_count = mesh_info->mesh->m_vertices.size();

	memcpy(pose, tracker->m_pose.data(), 16 * sizeof(float));

}


extern "C" __declspec(dllexport) void deleteContext(void* context)
{
	TrackerContext* tracker_context = static_cast<TrackerContext*>(context);
	tracker_context->rgb_recording.clear(); 
	tracker_context->depth_recording.clear();

	SAFE_DELETE( tracker_context->m_fusion); 
	SAFE_DELETE(tracker_context->m_tracker);
	SAFE_DELETE(tracker_context->m_videoStreamReader);

}