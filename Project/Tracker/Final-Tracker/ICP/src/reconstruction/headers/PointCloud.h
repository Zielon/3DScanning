#ifndef PROJECT_POINT_CLOUD_H
#define PROJECT_POINT_CLOUD_H

#include <vector>
#include "../../Eigen.h"
#include "SystemParameters.h"
#include <opencv2/core/mat.hpp>
#include "../sources/NearestNeighbor.hpp"
#include "../../files-manager/headers/DatasetManager.h"
#include "Mesh.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

enum FilterType { bilateral, median };

class PointCloud
{
public:

	PointCloud(SystemParameters camera_parameters, cv::Mat& depth, cv::Mat& rgb, int downsamplingFactor = 1);

	~PointCloud();

	const std::vector<Vector3f>& getPoints() const;

	const std::vector<Vector3f>& getNormals() const;

	const std::vector<Vector4uc>& getColors() const;

	int getClosestPoint(Vector3f grid_cell);

	static cv::Mat filterMap(cv::Mat map, FilterType filter_type, int diameter, float sigma);

	float getDepthImage(int x, int y) const;

	std::vector<Match> queryNearestNeighbor(std::vector<Vector3f> points);

	void transform(Matrix4f transformation);

	cv::Mat getNormalMap();

	Matrix4f m_pose_estimation = Matrix4f::Identity();
	SystemParameters m_camera_parameters;
	int m_current_width = 0;
	int m_current_height = 0;
	int m_downsampling_factor = 1;
	unsigned short* m_depth_points_icp;
	float* m_depth_points_fusion;

private:
	void transform(cv::Mat& depth_mat, cv::Mat& rgb_mat);

	NearestNeighborSearch* m_nearestNeighbor;
	std::thread* m_indexBuildingThread = nullptr;
	bool m_filtering = false;

	std::vector<Vector3f> m_points;
	std::vector<Vector3f> m_normals;
	std::vector<Vector4uc> m_color_points;
	std::vector<Vector3f> m_grid_normals;//Required to compute the normal map

	//Juan Test
	cv::Mat depth_map;
};

#endif
