#include "../headers/PointCloud.h"

PointCloud::PointCloud(CameraParameters camera_parameters, cv::Mat& depth, int step_size)
	: m_camera_parameters(camera_parameters), m_step_size(step_size){

	m_nearestNeighbor = new NearestNeighborSearchFlann();
	this->transform(depth);
}

PointCloud::PointCloud(const PointCloud& point_cloud){
	m_points = std::vector<Vector3f>(point_cloud.m_points);
	m_normals = std::vector<Vector3f>(point_cloud.m_normals);
	m_camera_parameters = point_cloud.m_camera_parameters;
	m_step_size = point_cloud.m_step_size;
	m_nearestNeighbor = point_cloud.m_nearestNeighbor;
}

PointCloud::~PointCloud(){
	SAFE_DELETE(m_nearestNeighbor);
}

std::vector<Vector3f>& PointCloud::getPoints(){
	return m_points;
}

std::vector<Vector3f>& PointCloud::getNormals(){
	return m_normals;
}

const std::vector<Vector3f>& PointCloud::getPoints() const{
	return m_points;
}

const std::vector<Vector3f>& PointCloud::getNormals() const{
	return m_normals;
}

void PointCloud::transform(cv::Mat& depth){

	Vector3f pixel_coords;

	const auto height = m_camera_parameters.m_image_height;
	const auto width = m_camera_parameters.m_image_width;

	auto temp_points = std::vector<Vector3f>(width * height);
	auto temp_normals = std::vector<Vector3f>(width * height);

	//Depth range check
	//float depth_min = std::numeric_limits<float>::infinity();
	//float depth_max = -1;

	#pragma omp parallel for
	for (auto y = 0; y < height; y++)
	{
		for (auto x = 0; x < width; x++)
		{
			const unsigned int idx = y * width + x;

			auto depth_val = depth.at<float>(y, x);

			//Depth range check
			//depth_min = std::min(depth_min, depth_val);
			//depth_max = std::max(depth_max, depth_val);

			if (depth_val > 0.0f)
			{
				// Back-projection to camera space.
				pixel_coords << (x - m_camera_parameters.m_cX) / m_camera_parameters.m_fovX *
					depth_val, (y - m_camera_parameters.m_cY) / m_camera_parameters.m_fovY * depth_val, depth_val;

				temp_points[idx] = pixel_coords;

				//colIdx = idx * 4 * sizeof(BYTE);
				//vertices[idx].color << colorMap[colIdx], colorMap[colIdx + 1], colorMap[colIdx + 2], colorMap[colIdx + 3];
			}
			else
			{
				temp_points[idx] = Vector3f(MINF, MINF, MINF);
			}
		}
	}

	#pragma omp parallel for
	for (auto y = 1; y < height - 1; y++)
	{
		for (auto x = 1; x < width - 1; x++)
		{
			const unsigned int idx = y * width + x;

			unsigned int b = (y - 1) * width + x;
			unsigned int t = (y + 1) * width + x;
			unsigned int l = y * width + x - 1;
			unsigned int r = y * width + x + 1;

			const Vector3f diffX = temp_points[r] - temp_points[l];
			const Vector3f diffY = temp_points[t] - temp_points[b];

			temp_normals[idx] = -diffX.cross(diffY);
			temp_normals[idx].normalize();
		}
	}

	// We set invalid normals for border regions.
	for (int u = 0; u < width; ++u)
	{
		temp_normals[u] = Vector3f(MINF, MINF, MINF);
		temp_normals[u + (height - 1) * width] = Vector3f(MINF, MINF, MINF);
	}

	for (int v = 0; v < height; ++v)
	{
		temp_normals[v * width] = Vector3f(MINF, MINF, MINF);
		temp_normals[(width - 1) + v * width] = Vector3f(MINF, MINF, MINF);
	}

	for (int i = 0; i < temp_points.size(); i += m_step_size)
	{
		const auto& point = temp_points[i];
		const auto& normal = temp_normals[i];

		if (point.allFinite() && normal.allFinite())
		{
			m_points.push_back(point);
			m_normals.push_back(normal);
		}
	}

	m_nearestNeighbor->buildIndex(m_points);
}

int PointCloud::getClosestPoint(Vector3f grid_cell) const{

	auto closestPoints = m_nearestNeighbor->queryMatches({grid_cell});

	if (!closestPoints.empty())
		return closestPoints[0].idx;

	return -1;
}
