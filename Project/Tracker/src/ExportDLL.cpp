#include "ExportDLL.h"
#include "marshaling/__Mesh.h"

extern "C" __declspec(dllexport) void* createContext(const char* dataset_path){

	auto* tracker_context = new TrackerContext();

	tracker_context->m_videoStreamReader = new DatasetVideoStreamReader(dataset_path, false);

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

	cv::Mat rgb, depth;

	tracker_context->m_videoStreamReader->getNextFrame(rgb, depth, false);

	PointCloud* _source = new PointCloud(tracker_context->m_tracker->getCameraParameters(), depth, rgb, 8);
	std::shared_ptr<PointCloud> source(_source);

	if (tracker_context->m_first_frame) // first frame
	{
		tracker_context->m_first_frame = false;
		tracker_context->m_tracker->m_previous_point_cloud = source;
		memcpy(pose, tracker_context->m_tracker->m_previous_pose.data(), 16 * sizeof(float));
		return;
	}

	const Matrix4f delta_pose = tracker_context->m_tracker->alignNewFrame(
		source, tracker_context->m_tracker->m_previous_point_cloud);

	tracker_context->m_tracker->m_previous_pose = tracker_context->m_tracker->m_previous_pose * delta_pose;

	memcpy(pose, tracker_context->m_tracker->m_previous_pose.data(), 16 * sizeof(float));

	source->m_pose_estimation = tracker_context->m_tracker->m_previous_pose;

	// Produce a new point cloud (add to the buffer)
	tracker_context->m_fusion->produce(std::shared_ptr<PointCloud>(tracker_context->m_tracker->m_previous_point_cloud));

	// Safe the last frame reference
	tracker_context->m_tracker->m_previous_point_cloud = source;

	//So turns out opencv actually uses bgr not rgb...
	//no more opencv computations after this point
	cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);
	std::memcpy(image, rgb.data, rgb.rows * rgb.cols * sizeof(unsigned char) * 3);
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
