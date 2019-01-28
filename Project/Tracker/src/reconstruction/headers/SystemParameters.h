#pragma once

struct SystemParameters
{
	SystemParameters() = default;

	SystemParameters(float focal_length_X, float focal_length_Y, float cx, float cy, int image_height,
	                 int image_width, Matrix3f intrinsics, int volume_size, float truncation_scaling) :
		m_focal_length_X(focal_length_X), m_focal_length_Y(focal_length_Y), m_cX(cx), m_cY(cy),
		m_image_height(image_height), m_image_width(image_width), m_intrinsics(intrinsics),
		m_volume_size(volume_size), m_truncation_scaling(truncation_scaling){ }

	float m_focal_length_X = 0;
	float m_focal_length_Y = 0;
	float m_cX = 0;
	float m_cY = 0;
	int m_image_height = 0;
	int m_image_width = 0;
	float m_depth_max = INFINITY;
	float m_depth_min = INFINITY;
	Matrix3f m_intrinsics;
	int m_volume_size = 128;
	float m_truncation_scaling = 5.f;
};
