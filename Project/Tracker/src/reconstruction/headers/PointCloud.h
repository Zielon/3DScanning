#ifndef PROJECT_POINT_CLOUD_H
#define PROJECT_POINT_CLOUD_H

#include <vector>
#include "../../Eigen.h"
#include "CameraParameters.h"
#include <opencv2/core/mat.hpp>
#include "../sources/NearestNeighbor.hpp"
#include "../../files-manager/headers/DatasetManager.h"
#include "Mesh.h"
#include <atomic>

#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2\highgui\highgui.hpp>

enum FilterType { bilateral, median};

class PointCloud
{
public:

	PointCloud(CameraParameters camera_parameters, cv::Mat& depth, cv::Mat& rgb, int downsamplingFactor = 1);

	~PointCloud();

	void save(std::string name);

	std::vector<Vector3f>& getPoints();

	std::vector<Vector3f>& getNormals();

	std::vector<Vector4uc>& getColors();

	const std::vector<Vector3f>& getPoints() const;

	const std::vector<Vector3f>& getNormals() const;

	const std::vector<Vector4uc>& getColors() const;

	int getClosestPoint(Vector3f grid_cell);

	cv::Mat filterMap(cv::Mat map, FilterType filter_type, int diameter, float sigma);

	float getDepthImage(int x, int y) const;

	std::vector<Match> queryNearestNeighbor(std::vector<Vector3f> points); 

	//Juan Test
	void transform(Matrix4f transformation);

	Matrix4f m_pose_estimation = Matrix4f::Identity();
	CameraParameters m_camera_parameters;
	Mesh m_mesh; // For testing purpose
	int m_current_width = 0;
	int m_current_height = 0;

	std::atomic<int> refCounter = 2; 

private:
	void transform(cv::Mat& depth_mat, cv::Mat& rgb_mat);

	NearestNeighborSearch* m_nearestNeighbor;
	std::thread* m_indexBuildingThread = nullptr; 
	int m_downsampling_factor = 1;
	bool m_filtering = true;

	std::vector<Vector3f> m_points;
	std::vector<Vector3f> m_normals;
	std::vector<Vector4uc> m_color_points;
	std::vector<float> m_depth_points;
};

#endif
