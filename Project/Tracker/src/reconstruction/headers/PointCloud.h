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

	PointCloud(CameraParameters camera_parameters, cv::Mat& depth, int step_size);

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

	Matrix4f m_pose_estimation;

private:
	void transform(cv::Mat& depth);

	NearestNeighborSearch* m_nearestNeighbor;
	CameraParameters m_camera_parameters;
	int m_step_size = 8;
	std::vector<Vector3f> m_points;
	std::vector<Vector3f> m_normals;
};

#endif
