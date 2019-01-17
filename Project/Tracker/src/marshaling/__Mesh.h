#pragma once

#include "../reconstruction/headers/Mesh.h"

struct __Mesh
{
	int m_vertex_float_count;
	int m_index_count;
	int* m_index_buffer;
	float* m_vertex_buffer;
};

struct __MeshInfo
{
	int m_vertex_count;
	int m_index_count;
	Mesh* mesh; 
};