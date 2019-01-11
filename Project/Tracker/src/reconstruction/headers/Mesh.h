#ifndef TRACKER_LIB_MESH_H
#define TRACKER_LIB_MESH_H

#include <vector>
#include <ostream>
#include <Eigen/StdVector>
#include <fstream>
#include <list>
#include <iostream>

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

	void Clear(){
		m_vertices.clear();
		m_triangles.clear();
	}

	unsigned int AddVertex(Eigen::Vector3f& vertex){
		unsigned int vId = (unsigned int)m_vertices.size();
		m_vertices.emplace_back(vertex);
		return vId;
	}

	unsigned int AddFace(unsigned int idx0, unsigned int idx1, unsigned int idx2){
		unsigned int fId = (unsigned int)m_triangles.size();
		Triangle triangle(idx0, idx1, idx2);
		m_triangles.emplace_back(triangle);
		return fId;
	}

	std::list<Eigen::Vector3f>& GetVertices(){
		return m_vertices;
	}

	std::list<Triangle>& GetTriangles(){
		return m_triangles;
	}

	bool WriteMesh(const std::string& filename){
		// Write off file
		std::ofstream out_file(filename + ".off");
		if (!out_file.is_open()) return false;

		// write header
		out_file << "OFF" << std::endl;
		out_file << m_vertices.size() << " " << m_triangles.size() << " 0" << std::endl;

		// save vertices
		for (auto vertex : m_vertices)
		{
			out_file << vertex.x() << " " << vertex.y() << " " << vertex.z() << std::endl;
		}

		// save faces
		for (auto triangle : m_triangles)
		{
			out_file << "3 " << triangle.idx0 << " " << triangle.idx1 << " " << triangle.idx2 << std::endl;
		}

		// close file
		out_file.close();

		return true;
	}

private:
	std::list<Eigen::Vector3f> m_vertices;
	std::list<Triangle> m_triangles;
};

#endif
