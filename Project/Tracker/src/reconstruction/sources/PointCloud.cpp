#include "../headers/PointCloud.h"
#include <opencv2/imgproc.hpp>

PointCloud::PointCloud(CameraParameters camera_parameters, cv::Mat& depth, cv::Mat& rgb, bool downsampling)
	: m_camera_parameters(camera_parameters){

	m_nearestNeighbor = new NearestNeighborSearchFlann();
	m_downsampling = downsampling;

	this->transform(depth, rgb);
}

PointCloud::~PointCloud(){
	SAFE_DELETE(m_nearestNeighbor);
}

void PointCloud::save(std::string name){ }

std::vector<Vector3f>& PointCloud::getPoints(){
	return m_points;
}

std::vector<Vector3f>& PointCloud::getNormals(){
	return m_normals;
}

std::vector<Vector4uc>& PointCloud::getColors(){
	return m_color_points;
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

	int idx = y * m_current_width + x;

	if (idx < m_depth_points.size())
		return m_depth_points[idx];

	return INFINITY;
}

/// Downsample image 1 time
void PointCloud::transform(cv::Mat& depth_mat, cv::Mat& rgb_mat){

	cv::Mat image, colors;

	// TODO Think about it!
	//if (m_downsampling)
	//{
	//	pyrDown(depth_mat, image, cv::Size(depth_mat.cols / 2, depth_mat.rows / 2));
	//	pyrDown(rgb_mat, colors, cv::Size(depth_mat.cols / 2, depth_mat.rows / 2));
	//}
	//else
	//{
	image = cv::Mat(depth_mat);
	colors = cv::Mat(rgb_mat);
	//}

	Vector3f pixel_coords;

	m_current_height = image.rows;
	m_current_width = image.cols;

	auto size = m_current_width * m_current_height;

	m_depth_points = std::vector<float>(size);
	m_color_points = std::vector<Vector4uc>(size);

	// Temp vector for filtering
	auto temp_points = std::vector<Vector3f>(size);
	auto temp_normals = std::vector<Vector3f>(size);

	//Depth range check
	float depth_min = std::numeric_limits<float>::infinity();
	float depth_max = -std::numeric_limits<float>::infinity();

	for (auto y = 0; y < m_current_height; y++)
	{
		for (auto x = 0; x < m_current_width; x++)
		{
			const unsigned int idx = y * m_current_width + x;

			float depth = image.at<float>(y, x);
			auto color = colors.at<cv::Vec3b>(y, x);

			m_depth_points[idx] = depth;
			m_color_points[idx] = Vector4uc(color[0], color[1], color[2], 0);

			//Depth range check
			depth_min = std::min(depth_min, depth);
			depth_max = std::max(depth_max, depth);

			if (depth > 0.0f)
			{
				// Back-projection to camera space.
				pixel_coords << (x - m_camera_parameters.m_cX) / m_camera_parameters.m_focal_length_X *
					depth, (y - m_camera_parameters.m_cY) / m_camera_parameters.m_focal_length_Y * depth,
					depth;

				temp_points[idx] = pixel_coords;
			}
			else
			{
				temp_points[idx] = Vector3f(MINF, MINF, MINF);
			}
		}
	}

	m_camera_parameters.m_depth_max = depth_max;
	m_camera_parameters.m_depth_min = depth_min;

	for (auto y = 1; y < m_current_height - 1; y++)
	{
		for (auto x = 1; x < m_current_width - 1; x++)
		{
			const unsigned int idx = y * m_current_width + x;

			unsigned int b = (y - 1) * m_current_width + x;
			unsigned int t = (y + 1) * m_current_width + x;
			unsigned int l = y * m_current_width + x - 1;
			unsigned int r = y * m_current_width + x + 1;

			const Vector3f diffX = temp_points[r] - temp_points[l];
			const Vector3f diffY = temp_points[t] - temp_points[b];

			temp_normals[idx] = -diffX.cross(diffY);
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

		if (point.allFinite() && normal.allFinite())
		{
			m_points.push_back(point);
			m_normals.push_back(normal);
		}
	}

	#ifdef TESTING
	// To build this mesh we need all points from the image
	// m_mesh = Mesh(temp_points, m_color_points, m_current_width, m_current_height);
	#endif

	//m_nearestNeighbor->buildIndex(m_points);
}

int PointCloud::getClosestPoint(Vector3f grid_cell) const{

	auto closestPoints = m_nearestNeighbor->queryMatches({grid_cell});

	if (!closestPoints.empty())
		return closestPoints[0].idx;

	return -1;
}
