#ifndef TRACKER_LIB_FUSION_H
#define TRACKER_LIB_FUSION_H

#include "../../Eigen.h"



/**
 * Volumetric m_fusion class
 */
class Fusion {
public:

	Fusion(int width, int height, int pixelSteps) : m_width(width), m_height(height), m_pixelSteps(pixelSteps)
	{

	}

    void integrate(const std::vector<Vector3f> &cloud, Matrix4f &pose);
	std::vector<int> m_currentIndexBuffer; 

private:
	int m_height, m_width, m_pixelSteps; 

};

#endif //TRACKER_LIB_FUSION_H
