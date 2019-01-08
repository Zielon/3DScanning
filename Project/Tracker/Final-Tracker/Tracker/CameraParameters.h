#pragma once

struct CameraParameters
{
	CameraParameters() = default;

	CameraParameters(float fovX, float fovY, float cx, float cy, int image_height, int image_width) :
		m_fovX(fovX), m_fovY(fovY), m_cX(cx), m_cY(cy),
		m_image_height(image_height), m_image_width(image_width) { }

	float m_fovX = 0;
	float m_fovY = 0;
	float m_cX = 0;
	float m_cY = 0;
	int m_image_height = 0;
	int m_image_width = 0;
};