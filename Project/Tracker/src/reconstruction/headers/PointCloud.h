#ifndef PROJECT_POINT_CLOUD_H
#define PROJECT_POINT_CLOUD_H

#include <vector>
#include "../../Eigen.h"
#include "CameraParameters.h"
#include <opencv2/core/mat.hpp>

class PointCloud
{
public:

	PointCloud();

	PointCloud(CameraParameters camera_parameters, cv::Mat& depth);

	std::vector<Vector3f>& getPoints();

	std::vector<Vector3f>& getNormals();

	const std::vector<Vector3f>& getPoints() const;

	const std::vector<Vector3f>& getNormals() const;

private:
	void transform(cv::Mat& depth);

	CameraParameters m_camera_parameters;
	std::vector<Vector3f> m_points;
	std::vector<Vector3f> m_normals;
};

#endif