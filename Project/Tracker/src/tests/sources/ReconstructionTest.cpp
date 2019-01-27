#include "../headers/ReconstructionTest.h"
#include "../../debugger/headers/ProgressBar.hpp"

void ReconstructionTest::pointCloudTest() const{

	Verbose::message("START streamPointCloudTest()");

	TrackerContext* context = static_cast<TrackerContext*>(createContext(DatasetManager::getCurrentPath().data()));

	auto* img = new unsigned char[getImageWidth(context) * getImageHeight(context) * 3];

	for (int index = 0; index < 600; index += 50)
	{
		const auto trajectory = getTrajectory(index);

		// Process each point cloud in a different thread
		ThreadManager::add([context, index, trajectory](){
			cv::Mat rgb, depth;
			dynamic_cast<DatasetVideoStreamReader*>(context->m_videoStreamReader)->readAnyFrame(index, rgb, depth);
			Mesh mesh(depth, rgb, context->m_tracker->getCameraParameters());
			mesh.transform(trajectory);
			mesh.save("point_cloud_" + std::to_string(index));
		});
	}

	ThreadManager::waitForAll();

	Verbose::message("DONE streamPointCloudTest()", SUCCESS);

	delete[]img;
	SAFE_DELETE(context);

}

void ReconstructionTest::unityIntegrationTest() const{
	Verbose::message("START unityIntegrationTest()");

	TrackerContext* context = static_cast<TrackerContext*>(createContext(DatasetManager::getCurrentPath().data()));

	auto* img = new unsigned char[getImageWidth(context) * getImageHeight(context) * 3];

	auto size = getIterations();

	float pose[16];
	std::chrono::high_resolution_clock::time_point t2;
	const int SAVE_MESH_INTERVAL = 200;

	double sum_track = 0.0;
	double sum_getMesh = 0.0;

	for (int index = 0; index < size; index += 1)
	{
		std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

		tracker(context, img, pose);

		t2 = std::chrono::high_resolution_clock::now();

		if (index % SAVE_MESH_INTERVAL == 0 || index == size - 1)
		{
			//			context->m_fusion->wait();

			__MeshInfo meshinfo;
			getMeshInfo(context, &meshinfo);

			assert(meshinfo.mesh->m_vertices.size() == meshinfo.m_vertex_count);
			assert(meshinfo.mesh->m_triangles.size() == meshinfo.m_index_count / 3);
			assert(meshinfo.m_index_count % 3 == 0);
			t2 = std::chrono::high_resolution_clock::now();

			meshinfo.mesh->save("mesh_" + std::to_string(index));
		}

		std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

		std::cout << "Frame_" << index << ": ";
		std::cout << std::setprecision(3) << time_span.count() * 1000 << "ms";

		if (index % SAVE_MESH_INTERVAL == 0 || index == size - 1)
		{
			std::cout << "  [mesh]";
			sum_getMesh += time_span.count();
		}
		else
		{
			sum_track += time_span.count();
		}
		std::cout << endl;
	}

	context->m_fusion->save("mesh_FINAL");

	int num_mesh_frames = std::ceil(1.0 * size / SAVE_MESH_INTERVAL);

	std::cout << "Average Time for tracking frame: " << sum_track / (size - num_mesh_frames) * 1000 << "ms" << std::
		endl;
	std::cout << "Average Time for getMesh frame:  " << sum_getMesh / num_mesh_frames * 1000 << "ms" << std::endl;

	Verbose::message("DONE unityIntegrationTest()", SUCCESS);

	delete[]img;
	SAFE_DELETE(context);
}

void ReconstructionTest::reconstructionTest(int skip, int subsampling) const{

	Verbose::message("START reconstructionTest()");

	TrackerContext* context = static_cast<TrackerContext*>(createContext(DatasetManager::getCurrentPath().data()));

	auto* img = new unsigned char[getImageWidth(context) * getImageHeight(context) * 3];

	auto size = getIterations();

	ProgressBar bar(size, 60, "Frames loaded");

	for (int index = 0; index < size; index += skip)
	{
		const auto trajectory = getTrajectory(index);
		cv::Mat rgb, depth;

		dynamic_cast<DatasetVideoStreamReader*>(context->m_videoStreamReader)->readAnyFrame(index, rgb, depth);

		PointCloud* _cloud = new PointCloud(context->m_tracker->getCameraParameters(), depth, rgb, 1);
		std::shared_ptr<PointCloud> cloud(_cloud);

		cloud->m_pose_estimation = trajectory;
		context->m_fusion->produce(cloud);

		// Waits for the index building thread to finish before deleting the point cloud
		//cloud->getClosestPoint(Vector3f::Zero());

		bar.set(index);
		bar.display();


	}

	bar.done();

	context->m_fusion->save("mesh");

	Verbose::message("DONE reconstructionTest()", SUCCESS);

	delete[]img;
	SAFE_DELETE(context);
}

void ReconstructionTest::reconstructionTestWithOurTracking(int skip) const{

	Verbose::message("START reconstructionTestWithOurTracking()");

	TrackerContext* context = static_cast<TrackerContext*>(createContext(DatasetManager::getCurrentPath().data()));

	auto* img = new unsigned char[getImageWidth(context) * getImageHeight(context) * 3];

	const auto size = getIterations();

	float pose[16];

	ProgressBar bar(size, 60, "Frames loaded");

	for (int index = 0; index < size; index += skip)
	{
		tracker(context, img, pose);

		Matrix4f mat = Matrix4f::Map(pose); 

	//	std::cout << "\n" << mat << "\n" << std::endl; 

		bar.set(index);
		bar.display();
	}

	bar.done();

	context->m_fusion->save("mesh");

	Verbose::message("DONE reconstructionTestWithOurTracking()", SUCCESS);

	delete[]img;
	SAFE_DELETE(context);
}

void ReconstructionTest::reconstructionTestSensor(int mesh_index) const{
	Verbose::message("START reconstructionTestSensor()");

	TrackerContext* context = static_cast<TrackerContext*>(createSensorContext());
	float pose[16];
	auto* img = new unsigned char[getImageWidth(context) * getImageHeight(context) * 3];
	int index = 0;

	while (!wasKeyboardHit())
	{
		tracker(context, img, pose);

		if (index % mesh_index == 0)
		{
			Mesh mesh;
			context->m_fusion->processMesh(mesh);
		}

		index++;
	
		printf("Frame %d processed\n", index);
	}

	context->m_fusion->save("mesh");

	Verbose::message("DONE reconstructionTestSensor()", SUCCESS);

	delete[]img;
	SAFE_DELETE(context);
}

void ReconstructionTest::pointCloudTestWithICP() const{

	Verbose::message("START pointCloudTestWithICP()");

	TrackerContext* context = static_cast<TrackerContext*>(createContext(DatasetManager::getCurrentPath().data()));

	for (int index = 0; index < getIterations(); index++)
	{
		cv::Mat rgb, depth;

		dynamic_cast<DatasetVideoStreamReader*>(context->m_videoStreamReader)->readAnyFrame(index, rgb, depth);

		PointCloud* _target = new PointCloud(context->m_tracker->getCameraParameters(), depth, rgb, 8);
		std::shared_ptr<PointCloud> data(_target);

		if (index == 0)
		{
			context->m_tracker->m_pose = Matrix4f::Identity();
			context->m_tracker->m_previous_point_cloud = data;
			continue;
		}

		Matrix4f delta = context->m_tracker->alignNewFrame(context->m_tracker->m_previous_point_cloud, data);
		context->m_tracker->m_pose *= delta;

		if (index % 50 == 0  || index == 1)
		{
			Mesh mesh(depth, rgb, context->m_tracker->getCameraParameters());
			mesh.transform(context->m_tracker->m_pose);
			mesh.save("point_cloud_" + std::to_string(index));
		}

		context->m_tracker->m_previous_point_cloud = data;
	}

	Verbose::message("DONE pointCloudTestWithICP()", SUCCESS);
	SAFE_DELETE(context);
}
