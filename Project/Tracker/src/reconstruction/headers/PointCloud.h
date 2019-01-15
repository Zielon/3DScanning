#ifndef PROJECT_POINT_CLOUD_H
#define PROJECT_POINT_CLOUD_H

#include <vector>
#include "../../Eigen.h"
#include "CameraParameters.h"
#include <opencv2/core/mat.hpp>
#include "../sources/NearestNeighbor.hpp"
#include "../../files-manager/headers/DatasetManager.h"
#include "Mesh.h"

class PointCloud
{
public:

	PointCloud(CameraParameters camera_parameters, cv::Mat& depth, cv::Mat& rgb, bool downsampling = true);

	~PointCloud();

	void save(std::string name);

	std::vector<Vector3f>& getPoints();

	std::vector<Vector3f>& getNormals();

	std::vector<Vector4uc>& getColors();

	const std::vector<Vector3f>& getPoints() const;

	const std::vector<Vector3f>& getNormals() const;

	const std::vector<Vector4uc>& getColors() const;

	int getClosestPoint(Vector3f grid_cell) const;

	float getDepthImage(int x, int y) const;

	//Juan Test
	void transform(Matrix4f transformation);

	Matrix4f m_pose_estimation = Matrix4f::Identity();
	NearestNeighborSearch* m_nearestNeighbor;
	CameraParameters m_camera_parameters;
	Mesh m_mesh;
	int m_current_width = 0;
	int m_current_height = 0;

private:
	void transform(cv::Mat& depth_mat, cv::Mat& rgb_mat);

	bool m_downsampling = true;
	std::vector<Vector3f> m_points;
	std::vector<Vector3f> m_normals;
	std::vector<Vector4uc> m_color_points;
	std::vector<float> m_depth_points;
};

#endif
