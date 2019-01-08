#ifndef TRACKER_LIB_FUSION_H
#define TRACKER_LIB_FUSION_H

#include "../../Eigen.h"

#define EDGE_THRESHOLD 0.1f // 10cm
#define EDGE_THRESHOLD_SQ EDGE_THRESHOLD*EDGE_THRESHOLD


/**
 * Volumetric fusion class
 */
class Fusion {
public:

	Fusion(int width, int height, int pixelSteps) : m_width(width), m_height(height), m_pixelSteps(pixelSteps)
	{

	}

    void integrate(const std::vector<Vector3f> &cloud, Matrix4f &pose);

	std::vector<int> m_currentFrameIndexBuffer; 

	void generateMeshFromVertices(const std::vector<Vector3f> &verts, std::vector<int>& outIndices);

	inline unsigned int WriteIfValidTriangle(std::vector<int>& outIndices, const Vector3f& v0, const Vector3f& v1,
		const Vector3f& v2, const size_t& i0, const size_t& i1, const size_t& i2, const float& edgeThresholdSQ);

private:
	int m_height, m_width, m_pixelSteps; 

};

#endif //TRACKER_LIB_FUSION_H
