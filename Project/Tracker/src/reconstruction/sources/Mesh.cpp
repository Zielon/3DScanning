#include "../headers/Mesh.h"
#include <direct.h>

Mesh::Mesh() = default;

Mesh::Mesh(const PointCloud* cloud){

	const float edge_threshold = 0.01f;
	auto vertices = cloud->getPoints();

	m_vertices.insert(m_vertices.end(), vertices.begin(), vertices.end());

	auto width = cloud->m_current_width;
	auto height = cloud->m_current_height;

	unsigned int idx0, idx1, idx2, idx3;
	Vector3f p0, p1, p2, p3;

	auto s = vertices.size() - 1;

	for (int y = 0; y < height - 1; y++)
		for (int x = 0; x < width - 1; x++)
		{
			float depth = cloud->getDepthImage(x, y);

			idx0 = y * width + x;
			idx1 = idx0 + 1;
			idx2 = idx0 + width;
			idx3 = idx2 + 1;

			if (idx0 > s || idx1 > s || idx2 > s || idx3 > s) continue;

			//Points
			p0 = vertices[idx0];
			p1 = vertices[idx1];
			p2 = vertices[idx2];
			p3 = vertices[idx3];

			//Upper Triangle
			if (isValidTriangle(p0, p2, p1, edge_threshold))
			{
				addFace(idx0, idx2, idx1);
			}

			//Bottom Triangle
			if (isValidTriangle(p2, p3, p1, edge_threshold))
			{
				addFace(idx2, idx3, idx1);
			}
		}
}

unsigned int Mesh::addVertex(Vector3f& vertex){
	auto v_id = static_cast<unsigned int>(m_vertices.size());
	m_vertices.emplace_back(vertex);
	return v_id;
}

unsigned Mesh::addFace(unsigned idx0, unsigned idx1, unsigned idx2){
	auto f_id = static_cast<unsigned int>(m_triangles.size());
	Triangle triangle(idx0, idx1, idx2);
	m_triangles.emplace_back(triangle);
	return f_id;
}

void Mesh::merge(const Mesh& mesh){
	m_triangles.insert(m_triangles.end(), mesh.m_triangles.begin(), mesh.m_triangles.end());
	m_vertices.insert(m_vertices.end(), mesh.m_vertices.begin(), mesh.m_vertices.end());
}

bool Mesh::save(const std::string& filename){
	std::string folder = "test_meshes";

	_mkdir(folder.data());

	std::ofstream out_file(folder + "\\" + filename + ".off");
	if (!out_file.is_open()) return false;

	std::cout << "Generating a mesh with " << m_vertices.size() << " vertices and " << m_triangles.size() <<
		" triangles" << std::endl;

	// write header
	out_file << "OFF" << std::endl;
	out_file << m_vertices.size() << " " << m_triangles.size() << " 0" << std::endl;

	// save vertices
	for (auto vertex : m_vertices)
	{
		auto position = (vertex.x() == MINF) ? Vector3f(0.0, 0.0, 0.0) : vertex;
		out_file << position.x() << " " << position.y() << " " << position.z() << std::endl;
	}

	// save faces
	for (auto triangle : m_triangles)
	{
		out_file << "3 " << triangle.idx0 << " " << triangle.idx1 << " " << triangle.idx2 << std::endl;
	}

	// close file
	out_file.close();

	std::cout << "Generating the mesh was successful!" << std::endl;

	return true;
}

bool Mesh::isValidTriangle(Vector3f p0, Vector3f p1, Vector3f p2, float edgeThreshold) const{
	if (p0.x() == MINF || p1.x() == MINF || p2.x() == MINF) return false;
	return !((p0 - p1).norm() >= edgeThreshold || (p1 - p2).norm() >= edgeThreshold ||
		(p2 - p1).norm() >= edgeThreshold);
}
