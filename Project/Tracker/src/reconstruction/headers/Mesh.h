#ifndef TRACKER_LIB_MESH_H
#define TRACKER_LIB_MESH_H

#include <vector>
#include <ostream>
#include <Eigen/StdVector>
#include <fstream>
#include <list>
#include <iostream>
#include "../../Eigen.h"
#include <thread>

struct Triangle
{
	unsigned int idx0;
	unsigned int idx1;
	unsigned int idx2;

	Triangle(unsigned int _idx0, unsigned int _idx1, unsigned int _idx2) :
		idx0(_idx0), idx1(_idx1), idx2(_idx2){}
};

class Mesh
{
public:

	Mesh();

	Mesh(std::vector<Vector3f>& vertices, std::vector<Vector4uc>& colors, int width, int height);

	unsigned int addVertex(Vector3f& vertex);

	unsigned int addFace(unsigned int idx0, unsigned int idx1, unsigned int idx2);

	void merge(const Mesh& mesh);

	bool save(const std::string& filename);

	void transform(const Matrix4f& trajectory);

	std::vector<Vector3f> m_vertices;
	std::vector<Triangle> m_triangles;
	std::vector<Vector4uc> m_colors;

private:
	bool isValidTriangle(Vector3f p0, Vector3f p1, Vector3f p2, float edgeThreshold) const;
};

#endif
