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

    [StructLayout(LayoutKind.Sequential)]
    public struct MeshDto
    {
        public int[] Triangles;
        public Vector3[] Vertices;
    }

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
    public struct __SystemParameters
    {
        public int m_volume_size;
        public float m_truncation_scaling;

        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 256)]
        public string m_dataset_path;
    }
}