#ifndef PROJECT_POINT_CLOUD_H
#define PROJECT_POINT_CLOUD_H

#include <vector>
#include "../../Eigen.h"
#include "CameraParameters.h"
#include <opencv2/core/mat.hpp>
#include "../sources/NearestNeighbor.hpp"

class PointCloud
{
public:

	PointCloud(CameraParameters camera_parameters, cv::Mat& depth);

	PointCloud(const PointCloud &point_cloud);

	~PointCloud();

	std::vector<Vector3f>& getPoints();

	std::vector<Vector3f>& getNormals();

	const std::vector<Vector3f>& getPoints() const;

	const std::vector<Vector3f>& getNormals() const;

	int getClosestPoint(Vector3f grid_cell) const;

	NearestNeighborSearch* getNearestNeighborSearch() const{
		return m_nearestNeighbor;
	};

	Matrix4f m_pose_estimation = Matrix4f::Identity();

	float depthImage(int x, int y);

private:
	void transform(cv::Mat& depth);

	cv::Mat m_depth;
	NearestNeighborSearch* m_nearestNeighbor;
	CameraParameters m_camera_parameters;
	std::vector<Vector3f> m_points;
	std::vector<Vector3f> m_normals;
};

#endif
