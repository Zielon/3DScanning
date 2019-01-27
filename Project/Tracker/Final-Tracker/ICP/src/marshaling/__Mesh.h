#pragma once

#include "../reconstruction/headers/Mesh.h"

struct __MeshInfo
{
	int m_vertex_count;
	int m_index_count;
	Mesh* mesh;
};
