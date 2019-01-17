using System;
using System.Runtime.InteropServices;

namespace Assets.Scripts
{
    [StructLayout(LayoutKind.Sequential)]
    public struct __Mesh
    {
        public int m_vertex_float_count;
        public int m_index_count;
        public IntPtr m_index_buffer;
        public IntPtr m_vertex_buffer;
    }


    [StructLayout(LayoutKind.Sequential)]
    public struct __MeshInfo
    {
        public int m_vertex_count;
        public int m_index_count;
        public IntPtr m_meshptr; 
    }

}