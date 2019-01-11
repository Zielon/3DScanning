#pragma once

struct CameraParameters
{
	CameraParameters() = default;

	CameraParameters(float focal_length_X, float focal_length_Y, float cx, float cy, int image_height,
	                 int image_width) :
		m_focal_length_X(focal_length_X), m_focal_length_Y(focal_length_Y), m_cX(cx), m_cY(cy),
		m_image_height(image_height), m_image_width(image_width){ }

	float m_focal_length_X = 0;
	float m_focal_length_Y = 0;
	float m_cX = 0;
	float m_cY = 0;
	int m_image_height = 0;
	int m_image_width = 0;
};
