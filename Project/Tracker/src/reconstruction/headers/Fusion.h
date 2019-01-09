#ifndef TRACKER_LIB_FUSION_H
#define TRACKER_LIB_FUSION_H

#include "../../Eigen.h"
#include "CameraParameters.h"

/**
 * Volumetric fusion class
 */
class Fusion
{
public:
	Fusion(CameraParameters camera_parameters): m_camera_parameters(camera_parameters){ }

	void integrate(const std::vector<Vector3f>& cloud, Matrix4f& pose);

private:
	CameraParameters m_camera_parameters;
};

#endif //TRACKER_LIB_FUSION_H
