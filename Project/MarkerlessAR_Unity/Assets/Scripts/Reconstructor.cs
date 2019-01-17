using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using UnityEngine;
using UnityEngine.UI;

namespace Assets.Scripts
{
    internal struct MeshDto
    {
        public int[] Triangles;
        public Vector3[] Vertices;
    }

    public class Reconstructor : MonoBehaviour
    {
        // Unity automatically find DLL files located on Assets/Plugins
        private const string DllFilePath = @"Tracker_release";
        private readonly Queue<MeshDto> _meshDtoQueue = new Queue<MeshDto>();
        private IntPtr _cppContext;
        private int _framesProcessed;
        private byte[] _image;
        private float[] _pose;
        private Thread _thread;
        private int _w = -1;
        private int _h = -1;

        // Unity injected vars
        public GameObject cameraRig;
        public GameObject frameMeshPrefab;

        [DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr createContext(byte[] path);

        [DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)]
        private static extern void tracker(IntPtr context, byte[] image, float[] pose);

        [DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)]
        private static extern int getImageWidth(IntPtr context);

        [DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)]
        private static extern int getImageHeight(IntPtr context);

        [DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)]
        private static extern void getMesh(IntPtr context, ref __Mesh mesh);

        // Use this for initialization
        private void Start()
        {
            var segments = new List<string>(
                    Application.dataPath.Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar))
                {"..", "Datasets", "freiburg", " "};

            var absolutePath = segments.Aggregate(
                (path, segment) => path += Path.AltDirectorySeparatorChar + segment).Trim();

            _cppContext = createContext(Encoding.ASCII.GetBytes(absolutePath));

            _w = getImageWidth(_cppContext);
            _h = getImageHeight(_cppContext);

            Debug.Log("Created Context. Image dimensions: " + _w + "x" + _h);

            _pose = new float[16];
            _image = new byte[_w * _h * 3];
        }

        // Update is called once per frame
        private void Update()
        {
            //   Debug.Log("Update test");

            tracker(_cppContext, _image, _pose);

            _framesProcessed++;

            //Create texture from image
            var tex = new Texture2D(_w, _h, TextureFormat.RGB24, false);

            tex.LoadRawTextureData(_image);
            tex.Apply();

            var videoBg = GetComponent<Image>();
            videoBg.sprite = Sprite.Create(tex, new Rect(0, 0, tex.width, tex.height), new Vector2(.5f, .5f));

            // Apply camera poses
            var pose = Helpers.GetPose(_pose);

            cameraRig.transform.position = pose.GetColumn(3) * 1000;
            cameraRig.transform.rotation = pose.rotation;

            //   Debug.Log("Pos: " + cameraRig.transform.position);
            //   Debug.Log("Rot: " + cameraRig.transform.rotation.eulerAngles);

            if (_meshDtoQueue.Count > 0)
                AddMesh(_meshDtoQueue.Dequeue());

            if (_framesProcessed % 60 != 0 || _thread != null && _thread.IsAlive) return;

            _thread = SpawnFrameMeshThread();
            _thread.Start();
        }

        private void AddMesh(MeshDto dto)
        {
            var mesh = new Mesh {vertices = dto.Vertices, triangles = dto.Triangles};

            mesh.RecalculateNormals();
            mesh.RecalculateBounds();

            var frameMeshObject = Instantiate(frameMeshPrefab);

            frameMeshPrefab.GetComponent<MeshFilter>().mesh = mesh;
        }

        private Thread SpawnFrameMeshThread()
        {
            return new Thread(() =>
            {
                var dllMesh = new __Mesh();
                getMesh(_cppContext, ref dllMesh);

                var indexBuffer = new int[dllMesh.m_vertex_count];
                var vertexBuffer = new float[dllMesh.m_vertex_count];

                Marshal.Copy(dllMesh.m_index_buffer, indexBuffer, 0, dllMesh.m_index_count);
                Marshal.Copy(dllMesh.m_vertex_buffer, vertexBuffer, 0, dllMesh.m_vertex_count);

                Debug.Log("Loaded mesh with " + vertexBuffer.Length + " vertices and " + vertexBuffer.Length +
                          " indices.");

                var vertices = new List<Vector3>();

                for (var i = 0; i < vertexBuffer.Length; i++)
                {
                    vertices.Add(new Vector3(vertexBuffer[i], vertexBuffer[i + 1], vertexBuffer[i + 2]));
                    i += 2;
                }

                _meshDtoQueue.Enqueue(new MeshDto
                {
                    Triangles = indexBuffer,
                    Vertices = vertices.ToArray()
                });
            });
        }
    }
}