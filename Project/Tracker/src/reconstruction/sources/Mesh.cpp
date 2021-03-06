#include "../headers/Mesh.h"
#include <direct.h>
#include "../../debugger/headers/Verbose.h"
#include <opencv2/core/mat.hpp>
#include "../../helpers/Transformations.h"

Mesh::Mesh() = default;

/// Create a mesh using naive approach to generate triangles
Mesh::Mesh(cv::Mat& depthMat, cv::Mat colorMat, SystemParameters system_parameters){

	const float edge_threshold = 0.01f;

	int height = depthMat.rows;
	int width = depthMat.cols;

	m_colors.resize(height * width);
	m_vertices.resize(height * width);

	#pragma omp parallel for
	for (auto y = 0; y < height; y++)
	{
		for (auto x = 0; x < width; x++)
		{
			const unsigned int idx = y * width + x;

			float depth = depthMat.at<float>(y, x);
			auto color = colorMat.at<cv::Vec3b>(y, x);

			m_colors[idx] = Vector4uc(color[0], color[1], color[2], 0);

			if (depth > 0.0f)
			{
				// Back-projection to camera space.
				m_vertices[idx] = Transformations::backproject(x, y, depth, system_parameters);
			}
			else
			{
				m_vertices[idx] = Vector3f(MINF, MINF, MINF);
			}
		}
	}

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
			p0 = m_vertices[idx0];
			p1 = m_vertices[idx1];
			p2 = m_vertices[idx2];
			p3 = m_vertices[idx3];

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
	m_mutex.lock();
	auto v_id = static_cast<unsigned int>(m_vertices.size());
	m_vertices.emplace_back(vertex);
	m_mutex.unlock();
	return v_id;
}

unsigned Mesh::addFace(unsigned idx0, unsigned idx1, unsigned idx2){
	m_mutex.lock();
	auto f_id = static_cast<unsigned int>(m_triangles.size());
	Triangle triangle(idx0, idx1, idx2);
	m_triangles.emplace_back(triangle);
	m_mutex.unlock();
	return f_id;
}

void Mesh::transform(const Matrix4f& matrix){
	// Camera space to world space
	for (int i = 0; i < m_vertices.size(); i++)
	{
		auto camera = m_vertices[i];
		if (camera.x() == MINF) continue;
		auto transform = matrix * Vector4f(camera[0], camera[1], camera[2], 1.f);
		m_vertices[i] = Vector3f(transform[0], transform[1], transform[2]);
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

	if (!m_colors.empty()) out_file << "COFF" << std::endl;
	else out_file << "OFF" << std::endl;

	out_file << m_vertices.size() << " " << m_triangles.size() << " 0" << std::endl;

	out_file << "# LIST OF VERTICES" << std::endl;
	out_file << "# X Y Z R G B A" << std::endl;

	// save vertices
	for (int i = 0; i < m_vertices.size(); i++)
	{
		auto vertex = m_vertices[i];
		auto color = m_colors[i];

		vertex = (vertex.x() == MINF) ? Vector3f(0.0, 0.0, 0.0) : vertex;
		out_file << vertex.x() << " " << vertex.y() << " " << vertex.z() << " ";

		if (!m_colors.empty())
		{
			auto color = m_colors[i];
			out_file << +color[0] << " " << +color[1] << " " << +color[2] << " " << 255 << std::endl;
		}
		else
			out_file << std::endl;
	}

	out_file << "# LIST OF FACES" << std::endl;

	// save faces
	for (auto triangle : m_triangles)
	{
		out_file << "3 " << triangle.idx0 << " " << triangle.idx1 << " " << triangle.idx2 << std::endl;
	}

	// close file
	out_file.close();

	Verbose::message("Generating the mesh was successful!", SUCCESS);

	return true;
}
