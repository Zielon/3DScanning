#include "../headers/PointCloud.h"
#include <opencv2/imgproc.hpp>
#include "../../helpers/Transformations.h"

PointCloud::PointCloud(SystemParameters camera_parameters, cv::Mat& depth, cv::Mat& rgb, int downsamplingFactor)
	: m_camera_parameters(camera_parameters){

	m_nearestNeighbor = new NearestNeighborSearchFlann();
	m_nearestNeighbor->setMatchingMaxDistance(max_distance);
	m_downsampling_factor = downsamplingFactor;

	this->transform(depth, rgb);
}

PointCloud::~PointCloud(){
	SAFE_DELETE(m_nearestNeighbor);
}

const std::vector<Vector3f>& PointCloud::getPoints() const{
	return m_points;
}

const std::vector<Vector3f>& PointCloud::getNormals() const{
	return m_normals;
}

const std::vector<Vector4uc>& PointCloud::getColors() const{
	return m_color_points;
}

float PointCloud::getDepthImage(int x, int y) const{

	if (x < 0 || y < 0 || x > m_current_width || y > m_current_height)
		return INFINITY;

	int idx = y * m_current_width + x;

	if (idx >= 0 && idx < m_current_width * m_current_height)
		return m_depth_points_fusion[idx];

	return INFINITY;
}

void PointCloud::transform(Matrix4f transformation){
	#pragma omp parallel for
	for (int i = 0; i < m_points.size(); i++)
	{
		auto camera = m_points[i];
		if (camera.x() == MINF) continue;
		auto transform = transformation * Vector4f(camera[0], camera[1], camera[2], 1.f);
		m_points[i] = Vector3f(transform[0], transform[1], transform[2]);
	}
}

cv::Mat PointCloud::getNormalMap(){
	//Normal computed directly by depth maps
	/*if (depth_map.type() != CV_32FC1) {
		depth_map.convertTo(depth_map, CV_32FC1);
	}

	cv::Mat normals(depth_map.size(), CV_32FC3);

	float depth_scale = 5000.0f;//This should be adapted depending of the sensor

	for (int y = 1; y < depth_map.rows - 1; y++)
	{
		for (int x = 1; x < depth_map.cols - 1; x++)
		{
			// 3d pixels
			  * * * * *
			  * * t * *
			  * l c * *
			  * * * * *

			cv::Vec3f t(x, y + 1, depth_scale * depth_map.at<float>(y + 1, x));
			cv::Vec3f l(x - 1, y, depth_scale * depth_map.at<float>(y, x - 1) );
			cv::Vec3f c(x, y, depth_scale * depth_map.at<float>(y, x) );
			cv::Vec3f d = (l - c).cross(t - c);
			
			normals.at<cv::Vec3f>(y, x) = normalize(d);
		}
	}*/

	cv::Mat normal_map(m_current_height, m_current_width, CV_8UC3, cv::Scalar(0, 0, 0));

	for (auto y = 0; y < m_current_height; y++)
	{
		for (auto x = 0; x < m_current_width; x++)
		{
			const unsigned int idx = y * m_current_width + x;

			//cv::Vec3f & color = normals.at<cv::Vec3f>(y, x);//Test

			cv::Vec3b& color = normal_map.at<cv::Vec3b>(y, x);

			auto normal = m_grid_normals[idx];

			//Unvalid normal -> black color
			if (!normal.allFinite())
			{
				color[0] = color[1] = color[2] = 0.0f;
				continue;
			}

			//Transforming normal to pixel color space
			normal = (0.5f * normal + Vector3f(0.5f, 0.5f, 0.5f)) * 255;

			color = cv::Vec3f(normal.x(), normal.y(), normal.z());
		}
	}

	//normals.convertTo(normal_map, CV_8UC3);
	//cv::imwrite("normal_map.png", normal_map);

	return normal_map;
}

/// Downsample image 1 time
void PointCloud::transform(cv::Mat& depth_mat, cv::Mat& rgb_mat){

	cv::Mat image, colors;

	//Juan Test
	depth_map = depth_mat.clone();

	if (m_downsampling_factor > 1)
	{
		resize(depth_mat, image,
		       cv::Size(depth_mat.cols / m_downsampling_factor, depth_mat.rows / m_downsampling_factor), 0, 0,
		       cv::INTER_NEAREST);
		resize(rgb_mat, colors,
		       cv::Size(depth_mat.cols / m_downsampling_factor, depth_mat.rows / m_downsampling_factor), 0, 0,
		       cv::INTER_NEAREST);
	}
	else
	{
		image = cv::Mat(depth_mat);
		colors = cv::Mat(rgb_mat);
	}

	m_current_height = image.rows;
	m_current_width = image.cols;

	auto size = m_current_width * m_current_height;

	m_depth_points_fusion = new float[size];
	m_depth_points_icp = new unsigned short[size];
	m_color_points = std::vector<Vector4uc>(size);

	// Temp vector for filtering
	auto temp_points = std::vector<Vector3f>(size);
	auto temp_normals = std::vector<Vector3f>(size);

	//Depth range check
	float depth_min = std::numeric_limits<float>::infinity();
	float depth_max = -std::numeric_limits<float>::infinity();

	//Bilateral filtering to remove noise
	cv::Mat filtered_depth;

	if (m_filtering)
	{
		FilterType filter_type = bilateral;
		filtered_depth = this->filterMap(depth_mat, filter_type, 9.0f, 150.0f);
	}

	#pragma omp parallel for
	for (auto y = 0; y < m_current_height; y++)
	{
		for (auto x = 0; x < m_current_width; x++)
		{
			const unsigned int idx = y * m_current_width + x;

			float depth = image.at<float>(y, x);
			auto color = colors.at<cv::Vec3b>(y, x);

			m_color_points[idx] = Vector4uc(color[0], color[1], color[2], 0);

			//ICP Cuda works in milimetres and Fusion in meters
			m_depth_points_icp[idx] = static_cast<unsigned short>(depth * 1000.f);
			m_depth_points_fusion[idx] = depth; //depth map already in meters

			//Depth range check
			depth_min = std::min(depth_min, depth);
			depth_max = std::max(depth_max, depth);

			if (depth > 0.0f)
			{
				// Back-projection to camera space.
				temp_points[idx] = Transformations::backproject(
					x * m_downsampling_factor,
					y * m_downsampling_factor,
					depth, m_camera_parameters);
			}
			else
			{
				temp_points[idx] = Vector3f(MINF, MINF, MINF);
			}
		}
	}

	m_camera_parameters.m_depth_max = depth_max;
	m_camera_parameters.m_depth_min = depth_min;

	#pragma omp parallel for
	for (auto y = 1; y < m_current_height - 1; y++)
	{
		for (auto x = 1; x < m_current_width - 1; x++)
		{
			const unsigned int idx = y * m_current_width + x;

			unsigned int b = (y - 1) * m_current_width + x;
			unsigned int t = (y + 1) * m_current_width + x;
			unsigned int l = y * m_current_width + x - 1;
			unsigned int r = y * m_current_width + x + 1;

			//Exercise 3 Formula
			const Vector3f diffX = temp_points[r] - temp_points[l];
			const Vector3f diffY = temp_points[t] - temp_points[b];

			temp_normals[idx] = -diffX.cross(diffY);

			//Kinect Fusion paper formula
			/*Vector3f diffX = temp_points[l] - temp_points[idx];
			Vector3f diffY = temp_points[b] - temp_points[idx];
			Vector3f d = -diffX.cross(diffY);

			temp_normals[idx] = d;*/

			temp_normals[idx].normalize();
		}
	}

	// We set invalid normals for border regions.
	for (int u = 0; u < m_current_width; ++u)
	{
		temp_normals[u] = Vector3f(MINF, MINF, MINF);
		temp_normals[u + (m_current_height - 1) * m_current_width] = Vector3f(MINF, MINF, MINF);
	}

	for (int v = 0; v < m_current_height; ++v)
	{
		temp_normals[v * m_current_width] = Vector3f(MINF, MINF, MINF);
		temp_normals[(m_current_width - 1) + v * m_current_width] = Vector3f(MINF, MINF, MINF);
	}

	for (int i = 0; i < temp_points.size(); i++)
	{
		const auto& point = temp_points[i];
		const auto& normal = temp_normals[i];

		m_grid_normals.push_back(normal);

		//FIX This part destroy the grid property of the point cloud
		if (point.allFinite() && normal.allFinite())
		{
			m_points.push_back(point);
			m_normals.push_back(normal);
		}
	}

	//m_indexBuildingThread = new std::thread([this]()-> void{
	//	m_nearestNeighbor->buildIndex(m_points);
	//});

}

std::vector<Match> PointCloud::queryNearestNeighbor(std::vector<Vector3f> points){
	if (m_indexBuildingThread != nullptr)
	{
		m_indexBuildingThread->join();
		SAFE_DELETE(m_indexBuildingThread);
	}

	return m_nearestNeighbor->queryMatches(points);
}

int PointCloud::getClosestPoint(Vector3f grid_cell){

	if (m_indexBuildingThread != nullptr)
	{
		m_indexBuildingThread->join();
		SAFE_DELETE(m_indexBuildingThread);
	}

	auto closestPoints = m_nearestNeighbor->queryMatches({grid_cell});

	if (!closestPoints.empty())
		return closestPoints[0].idx;

	return -1;
}

cv::Mat PointCloud::filterMap(cv::Mat map, FilterType filter_type, int diameter, float sigma){
	cv::Mat result;

	switch (filter_type)
	{
	case bilateral:
		bilateralFilter(map, result, diameter, sigma, sigma);
		break;

	case median:
		medianBlur(map, result, diameter);
		break;

	default: result = map;
		break;
	}

	return result;
}
