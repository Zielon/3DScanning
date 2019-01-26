#pragma once

#include <iostream>

#include "../../helpers/Transformations.h"
#include "../../Eigen.h"
#include "SystemParameters.h"
#include "Voxel.h"
#include "PointCloud.h"
#include "Volume.h"


struct FrustumBox
{
	Vector3i m_max;
	Vector3i m_min;
};



class FusionBase
{
public:
	FusionBase(SystemParameters camera_parameters) : m_camera_parameters(std::move(camera_parameters)) {};

	virtual ~FusionBase() {};

	virtual void consume() =0;

	virtual void produce(std::shared_ptr<PointCloud> cloud) = 0;

	virtual void integrate(std::shared_ptr<PointCloud> cloud) = 0;

	virtual void save(std::string name) = 0;

	virtual void processMesh(Mesh& mesh) = 0;

	virtual void wait() const = 0;

protected: 

	Vector3i clamp(Vector3i value) const {
		return value.cwiseMin(m_volume->m_size).cwiseMax(0);
	}


	FrustumBox computeFrustumBounds(Matrix4f cameraToWorld, SystemParameters camera_parameters) const
	{
		const auto rotation = cameraToWorld.block(0, 0, 3, 3);
		const auto translation = cameraToWorld.block(0, 3, 3, 1);

		auto width = camera_parameters.m_image_width;
		auto height = camera_parameters.m_image_height;
		auto min_depth = camera_parameters.m_depth_min;
		auto max_depth = camera_parameters.m_depth_max;

		std::vector<Vector3f> corners;

		// Image -> Camera -> World -> Grid
		for (auto depth : std::vector<float>{ min_depth, max_depth })
		{
			corners.push_back(Transformations::backproject(0, 0, depth, camera_parameters));
			corners.push_back(Transformations::backproject(width - 1, 0, depth, camera_parameters));
			corners.push_back(Transformations::backproject(width - 1, height - 1, depth, camera_parameters));
			corners.push_back(Transformations::backproject(0, height - 1, depth, camera_parameters));
		}

		Vector3i min;
		Vector3i max;

		for (int i = 0; i < 8; i++)
		{
			auto grid = m_volume->getGridPosition(rotation * corners[i] + translation);
			min = min.cwiseMin(grid).eval();
			max = max.cwiseMax(grid).eval();
		}

		FrustumBox box;

		box.m_min = clamp(min);
		box.m_max = clamp(max);

		return box;
	}

	Volume* m_volume;

	float m_trunaction = 0;

	SystemParameters m_camera_parameters;


};