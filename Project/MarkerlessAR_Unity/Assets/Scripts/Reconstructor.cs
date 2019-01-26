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
        private int _h = -1;
        private byte[] _image;
        private float[] _pose;
        private Thread _thread;

        //general setup
        private readonly bool _use_sensor = true;
        private int _w = -1;

        public int abortAfterNFrames = -1;
        //bool _use_reconstruction = true;
        //bool _use_fusion = true;

        // Unity injected vars
        public GameObject cameraRig;

        public GameObject frameMeshObject;
        public int meshUpdateRate = 2;
        public Image videoBG;

        [DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr createContext(byte[] path);

        [DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr createSensorContext();

        [DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)]
        private static extern void tracker(IntPtr context, byte[] image, float[] pose);

        [DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)]
        private static extern int getImageWidth(IntPtr context);

        [DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)]
        private static extern int getImageHeight(IntPtr context);

        [DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)]
        private static extern void getMeshInfo(IntPtr context, ref __MeshInfo mesh);

        [DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)]
        private static extern void getMeshBuffers(ref __MeshInfo mesh, Vector3[] vertices, int[] indices);


        // Use this for initialization
        private void Start()
        {
            if (_use_sensor)
            {
                _cppContext = createSensorContext();
            }
            else
            {
                var segments = new List<string>(
                        Application.dataPath.Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar))
                    {"..", "Datasets", "freiburg", " "};

                var absolutePath = segments.Aggregate(
                    (path, segment) => path += Path.AltDirectorySeparatorChar + segment).Trim();

                _cppContext = createContext(Encoding.ASCII.GetBytes(absolutePath));
            }

            _w = getImageWidth(_cppContext);
            _h = getImageHeight(_cppContext);

            Debug.Log("Created Context. Image dimensions: " + _w + "x" + _h);

            _pose = new float[16];
            _image = new byte[_w * _h * 3];
        }

        // Update is called once per frame
        private void Update()
        {
            // Unity just dies if the dataset runs out
            if (_framesProcessed > abortAfterNFrames && abortAfterNFrames > 0)
            {
                Debug.Log("Tracking aborted");
                return;
            }

            if (_w <= 0 || _h <= 0)
            {
                Debug.Log("There is a problem with the stream of the tracker [Check _use_sensor flag]");
                return;
            }

            tracker(_cppContext, _image, _pose);

            _framesProcessed++;

            //Create texture from image
            var tex = new Texture2D(_w, _h, TextureFormat.RGB24, false);

            tex.LoadRawTextureData(_image);
            tex.Apply();

            videoBG.sprite = Sprite.Create(tex, new Rect(0, 0, tex.width, tex.height), new Vector2(.5f, .5f));

            // Apply camera poses
            var pose = Helpers.GetPose(_pose);

            cameraRig.transform.position = pose.GetColumn(3);
            cameraRig.transform.rotation = pose.rotation;

            //   Debug.Log("Pos: " + cameraRig.transform.position);
            //   Debug.Log("Rot: " + cameraRig.transform.rotation.eulerAngles);

            if (_meshDtoQueue.Count > 0)
                AddMesh(_meshDtoQueue.Dequeue());

            //get first mesh after n frames
            if (_framesProcessed % meshUpdateRate != 0 || _thread != null && _thread.IsAlive) return;

            _thread = SpawnFrameMeshThread();
            _thread.Start();
        }

        private void AddMesh(MeshDto dto)
        {
            var mesh = new Mesh {vertices = dto.Vertices, triangles = dto.Triangles};

            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            frameMeshObject.GetComponent<MeshFilter>().mesh = mesh;
            frameMeshObject.GetComponent<MeshCollider>().sharedMesh = mesh;
        }

        private Thread SpawnFrameMeshThread()
        {
            return new Thread(() =>
            {
                var meshInfo = new __MeshInfo();
                getMeshInfo(_cppContext, ref meshInfo);

                var vertexBuffer = new Vector3[meshInfo.m_vertex_count];
                var indexBuffer = new int[meshInfo.m_index_count];

                getMeshBuffers(ref meshInfo, vertexBuffer, indexBuffer);
                Debug.Log("Loaded mesh with " + vertexBuffer.Length + " vertices and " + indexBuffer.Length +
                          " indices.");

                _meshDtoQueue.Enqueue(new MeshDto
                {
                    Triangles = indexBuffer,
                    Vertices = vertexBuffer
                });
            });
        }
    }
}