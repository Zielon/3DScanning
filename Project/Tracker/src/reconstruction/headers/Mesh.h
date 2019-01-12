#ifndef TRACKER_LIB_MESH_H
#define TRACKER_LIB_MESH_H

#include <vector>
#include <ostream>
#include <Eigen/StdVector>
#include <fstream>
#include <list>
#include <iostream>
#include "PointCloud.h"

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

	Mesh(const PointCloud* cloud);

	unsigned int addVertex(Vector3f& vertex);

	unsigned int addFace(unsigned int idx0, unsigned int idx1, unsigned int idx2);

	void merge(const Mesh& mesh);

	bool save(const std::string& filename);

	bool isValidTriangle(Vector3f p0, Vector3f p1, Vector3f p2, float edgeThreshold) const;

	std::list<Vector3f>& getVertices(){
		return m_vertices;
	}

	std::list<Triangle>& getTriangles(){
		return m_triangles;
	}

	void clear(){
		m_vertices.clear();
		m_triangles.clear();
	}

private:
	std::list<Vector3f> m_vertices;
	std::list<Triangle> m_triangles;
};

#endif
