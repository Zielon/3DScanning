using System;
using System.Runtime.InteropServices;
using UnityEngine; 

namespace Assets.Scripts
{
    [StructLayout(LayoutKind.Sequential)]
    public struct __MeshInfo
    {
        public int m_vertex_count;
        public int m_index_count;
        public IntPtr m_meshptr;
    }

    internal struct MeshDto
    {
        public int[] Triangles;
        public Vector3[] Vertices;
    }

}