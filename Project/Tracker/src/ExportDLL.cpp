#include "ExportDLL.h"
#include "marshaling/__Mesh.h"

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
	//std::cout << "Test Tracker Camera Pose" << std::endl;

	std::cout << "Previous Pose" << std::endl;
	std::cout << tracker_context->m_tracker->m_previous_pose << std::endl;

	Matrix4f deltaPose = tracker_context->m_tracker->alignNewFrame(
		source, tracker_context->m_tracker->m_previous_point_cloud);

	//tracker_context->m_tracker->m_previous_pose = deltaPose * tracker_context->m_tracker->m_previous_pose;
	tracker_context->m_tracker->m_previous_pose = tracker_context->m_tracker->m_previous_pose * deltaPose;

	std::cout << "Delta Pose" << std::endl;
	std::cout << deltaPose << std::endl;

	memcpy(pose, tracker_context->m_tracker->m_previous_pose.data(), 16 * sizeof(float));

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

	Matrix4f deltaPose = Matrix4f::Identity();

	//Matrix4f deltaPose = tracker_context->m_tracker->alignNewFrame(
	//	source, tracker_context->m_tracker->m_previous_point_cloud);

	tracker_context->m_tracker->m_previous_pose = deltaPose * tracker_context->m_tracker->m_previous_pose;

	// Produce a new point cloud (add to the buffer)
	tracker_context->m_fusion->produce(tracker_context->m_tracker->m_previous_point_cloud);

	// Safe the last frame reference
	tracker_context->m_tracker->m_previous_point_cloud = source;

	//So turns out opencv actually uses bgr not rgb...
	//no more opencv computations after this point
	cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);
	std::memcpy(image, rgb.data, rgb.rows * rgb.cols * sizeof(unsigned char) * 3);
}

extern "C" __declspec(dllexport) void getMesh(void* context, __Mesh* unity_mesh){
	TrackerContext* tracker_context = static_cast<TrackerContext*>(context);

	Mesh mesh;
	tracker_context->m_fusion->processMesh(mesh);

	vector<int> index_buffer;
	vector<float> vertex_buffer;

	for (auto triangle : mesh.m_triangles)
	{
		index_buffer.push_back(triangle.idx0);
		index_buffer.push_back(triangle.idx1);
		index_buffer.push_back(triangle.idx2);
	}

	for (auto vector : mesh.m_vertices)
	{
		vertex_buffer.push_back(vector.x());
		vertex_buffer.push_back(vector.y());
		vertex_buffer.push_back(vector.z());
	}

	unity_mesh->m_vertex_count = vertex_buffer.size();
	unity_mesh->m_index_count = index_buffer.size();
	unity_mesh->m_vertex_buffer = &vertex_buffer[0];
	unity_mesh->m_index_buffer = &index_buffer[0];
}

void getNormalBuffer(void* context, float* normals){
	TrackerContext* c = static_cast<TrackerContext*>(context);
	memcpy(normals, c->m_tracker->m_previous_point_cloud->getNormals().data(),
	       c->m_tracker->m_previous_point_cloud->getNormals().size() * sizeof(Vector3f));
}
