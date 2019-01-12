#include "../headers/Mesh.h"
#include <direct.h>
#include "../../debugger/headers/Verbose.h"

Mesh::Mesh() = default;

/// Create a mesh using naive approach to generate triangles
Mesh::Mesh(std::vector<Vector3f>& vertices, std::vector<Vector4uc>& colors, int width, int height){

	const float edge_threshold = 0.01f;

	m_vertices.insert(m_vertices.end(), vertices.begin(), vertices.end());
	m_colors.insert(m_colors.end(), colors.begin(), colors.end());

	unsigned int idx0, idx1, idx2, idx3;
	Vector3f p0, p1, p2, p3;

	for (int y = 0; y < height - 1; y++)
		for (int x = 0; x < width - 1; x++)
		{
			idx0 = y * width + x;
			idx1 = idx0 + 1;
			idx2 = idx0 + width;
			idx3 = idx2 + 1;

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

void Mesh::transform(const Matrix4f& matrix){
	// Camera space to world space
	for (int i = 0; i < m_vertices.size(); i++)
	{
		auto camera = m_vertices[i];
		if(camera.x() == MINF) continue;
		auto world = matrix * Vector4f(camera[0], camera[1], camera[2], 1.f);
		m_vertices[i] = Vector3f(world[0], world[1], world[2]);
	}
}

bool Mesh::isValidTriangle(Vector3f p0, Vector3f p1, Vector3f p2, float edgeThreshold) const{
	if (p0.x() == MINF || p1.x() == MINF || p2.x() == MINF) return false;
	return !((p0 - p1).norm() >= edgeThreshold || (p1 - p2).norm() >= edgeThreshold ||
		(p2 - p1).norm() >= edgeThreshold);
}

bool Mesh::save(const std::string& filename){
	std::string folder = "test_meshes";

	_mkdir(folder.data());

	std::ofstream out_file(folder + "\\" + filename + ".off");
	if (!out_file.is_open()) return false;

	Verbose::message(
		"Generating a mesh with " + std::to_string(m_vertices.size()) + " vertices and " + std::
		to_string(m_triangles.size()) + " triangles");

	// write header
	out_file << "COFF" << std::endl;
	out_file << m_vertices.size() << " " << m_triangles.size() << " 0" << std::endl;

	out_file << "# LIST OF VERTICES" << std::endl;
	out_file << "# X Y Z R G B A" << std::endl;

	// save vertices
	for (int i = 0; i < m_vertices.size(); i++)
	{
		auto vertex = m_vertices[i];
		auto color = m_colors[i];
		vertex = (vertex.x() == MINF) ? Vector3f(0.0, 0.0, 0.0) : vertex;
		out_file << vertex.x() << " " << vertex.y() << " " << vertex.z() << " " << std::endl;
		out_file << +color[0] << " " << +color[1] << " " << +color[2] << " " << 1 << std::endl;
	}

	out_file << "# LIST OF FACES" << std::endl;

	// save faces
	for (auto triangle : m_triangles)
	{
		out_file << "3 " << triangle.idx0 << " " << triangle.idx1 << " " << triangle.idx2 << std::endl;
	}

	// close file
	out_file.close();

	Verbose::message("Generating the mesh was successful!");

	return true;
}
