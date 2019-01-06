#include "../headers/Fusion.h"

void Fusion::integrate(const std::vector<Vector3f> &cloud, Matrix4f &pose) {

}

void Fusion::generateMeshFromVertices(const std::vector<Vector3f>& verts, std::vector<int>& outIndices)
{


	size_t i0, i1, i2, i3;

	size_t nFaces = 0; 

	for (size_t y = 0; y < m_height/m_pixelSteps - 1; ++y)
	{
		for (size_t x = 0; x < m_width/m_pixelSteps - 1; ++x)// compute max. 2 tris per 2x2 grid cell 
											// v0 -- v1
											// |     |
											// v2 -- v3
		{
			i0 = x + y * m_width / m_pixelSteps;
			i1 = i0 + 1;
			i2 = i0 + m_width / m_pixelSteps;
			i3 = i2 + 1;

			const Vector3f& v0 = verts[i0];
			const Vector3f& v1 = verts[i1];
			const Vector3f& v2 = verts[i2];
			const Vector3f& v3 = verts[i3];

			nFaces += WriteIfValidTriangle(outIndices, v0, v2, v1, i0, i2, i1, EDGE_THRESHOLD_SQ);
			nFaces += WriteIfValidTriangle(outIndices, v2, v3, v1, i2, i3, i1, EDGE_THRESHOLD_SQ);

		}
	}

}


unsigned int Fusion::WriteIfValidTriangle(std::vector<int>& outIndices, const Vector3f& v0, const Vector3f& v1,
	const Vector3f& v2, const size_t& i0, const size_t& i1, const size_t& i2, const float& edgeThresholdSQ)
{
	if (v0.isZero() || v1.isZero() || v2.isZero() 		
		|| (v0 - v1).squaredNorm() > edgeThresholdSQ
		|| (v0 - v1).squaredNorm() > edgeThresholdSQ
		|| (v1 - v2).squaredNorm() > edgeThresholdSQ)
		return 0;


	outIndices.push_back(i0); 
	outIndices.push_back(i1); 
	outIndices.push_back(i2); 
	return 1;
}
