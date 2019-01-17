#ifndef TRACKER_LIB_TRANSFORMATIONS_H
#define TRACKER_LIB_TRANSFORMATIONS_H

#include "../Eigen.h"
#include "../reconstruction/headers/CameraParameters.h"
#include "../marshaling/__Mesh.h"

class Transformations
{
public:

	static Vector3f backproject(float x, float y, float depth, CameraParameters camera_parameters){
		Vector3f point;
		point[0] = (x - camera_parameters.m_cX) / camera_parameters.m_focal_length_X * depth;
		point[0] = (y - camera_parameters.m_cY) / camera_parameters.m_focal_length_Y * depth;
		point[0] = depth;
		return point;
	}

	static Vector3f reproject(Vector3f point, CameraParameters camera_parameters){
		float x = camera_parameters.m_focal_length_X * point.x() / point.z() + camera_parameters.m_cX;
		float y = camera_parameters.m_focal_length_Y * point.y() / point.z() + camera_parameters.m_cY;
		return Vector3f(x, y, point.z());
	}

	static void transformMesh(__Mesh* __mesh, Mesh& mesh){
		std::vector<int> index_buffer;
		std::vector<float> vertex_buffer;

		for (auto triangle : mesh.m_triangles)
		{
			index_buffer.push_back(triangle.idx0);
			index_buffer.push_back(triangle.idx1);
			index_buffer.push_back(triangle.idx2);
		}

		for (auto vector : mesh.m_vertices)
		{
			vertex_buffer.push_back(vector.x());
			vertex_buffer.push_back(vector.y());
			vertex_buffer.push_back(vector.z());
		}

		__mesh->m_vertex_count = vertex_buffer.size();
		__mesh->m_index_count = index_buffer.size();
		__mesh->m_vertex_buffer = &vertex_buffer[0];
		__mesh->m_index_buffer = &index_buffer[0];
	}
};

#endif
