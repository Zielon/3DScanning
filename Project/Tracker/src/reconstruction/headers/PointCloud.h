#ifndef PROJECT_POINT_CLOUD_H
#define PROJECT_POINT_CLOUD_H

#include <vector>
#include <unordered_map>
#include "../../Eigen.h"
#include "CameraParameters.h"
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

	PointCloud(CameraParameters camera_parameters, cv::Mat& depth, cv::Mat& rgb, int m_point_subsampling_factor = 8);

	~PointCloud();

	const std::vector<Vector3f>& getPoints()const;

	const std::vector<Vector3f>& getNormals()const;

	const std::vector<Vector4uc>& getColors()const;

	int getClosestPoint(Vector3f grid_cell);

	static cv::Mat filterMap(cv::Mat map, FilterType filter_type, int diameter, float sigma);

	const cv::Mat& getDepthImage() const;

	std::vector<Match> queryNearestNeighbor(std::vector<Vector3f> points);

	void transform(Matrix4f transformation);

	Matrix4f m_pose_estimation = Matrix4f::Identity();
	CameraParameters m_camera_parameters;





private:
	void transform(cv::Mat& depth_mat, cv::Mat& rgb_mat);

	cv::Mat m_rgb_image; 
	cv::Mat m_depth_image; 

	NearestNeighborSearch* m_nearestNeighbor;
	std::thread* m_indexBuildingThread = nullptr;
	bool m_filtering = true;

	int m_point_subsampling_factor;
	int m_current_width = 0;
	int m_current_height = 0;

	std::vector<Vector3f> m_points;
	std::vector<Vector3f> m_normals;
	std::vector<Vector4uc> m_color_points;
};

#endif
