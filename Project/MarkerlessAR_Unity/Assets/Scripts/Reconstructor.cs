﻿using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading;
using UnityEngine;
using UnityEngine.UI;

namespace Assets.Scripts
{
    public class Reconstructor : MonoBehaviour
    {
        // Unity automatically find DLL files located on Assets/Plugins
        private const string DllFilePath = @"Tracker_release";
        private readonly Queue<__MeshDto> _meshDtoQueue = new Queue<__MeshDto>();

        //general setup
        private IntPtr _cppContext;
        private int _framesProcessed;
        private int _h = -1;
        private byte[] _image;
        private float[] _pose;
        private Thread _thread;
        public bool _use_sensor;
        private int _w = -1;

        public int abortAfterNFrames = -1;
        //bool _use_reconstruction = true;
        //bool _use_fusion = true;

        // Unity injected vars
        public GameObject cameraRig;

        public GameObject frameMeshObject;
        public int meshUpdateRate = 2;
        public Image videoBG;

        public void Exit()
        {
            // WARNING!
            // Delete context throws an error in the cpp code in the FusionGPU destructor
            // At this moment in the build mode we cannot properly destroy this scene!
            deleteContext(_cppContext);
            Debug.Log("The current Tracker context has been deleted!");
        }

        void OnDestroy()
        {
            Exit();
        }

        [DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr createContext(ref __SystemParameters param);

        [DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr createSensorContext(ref __SystemParameters param);

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

        [DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)]
        private static extern void deleteContext(IntPtr context);

        // Use this for initialization
        private void Start()
        {
            Debug.Log("Basic Physics scene");

            meshUpdateRate = PlayerPrefs.GetInt(Settings.MESH_UPDATE);
            _use_sensor = bool.Parse(PlayerPrefs.GetString(Settings.USE_SENSOR));

            var param = new __SystemParameters
            {
                m_dataset_path = PlayerPrefs.GetString(Settings.DATASET_PATH),
                m_truncation = PlayerPrefs.GetFloat(Settings.TRUNCATION),
                m_volume_size = PlayerPrefs.GetInt(Settings.VOLUME_SIZE),
                m_max_depth = PlayerPrefs.GetFloat(Settings.MAX_DEPTH)
            };

            _cppContext = _use_sensor ? createSensorContext(ref param) : createContext(ref param);

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
            if (!_use_sensor & (_framesProcessed > abortAfterNFrames) && abortAfterNFrames > 0 && !_use_sensor)
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

            if (_meshDtoQueue.Count > 0)
                AddMesh(_meshDtoQueue.Dequeue());

            if (_framesProcessed % meshUpdateRate != 0) return;

            LoadMesh();
        }

        private void AddMesh(__MeshDto dto)
        {
            var mesh = frameMeshObject.GetComponent<MeshFilter>().mesh;
            mesh.Clear();
            mesh.vertices = dto.Vertices;
            mesh.SetIndices(dto.Triangles, MeshTopology.Triangles, 0, true);
            mesh.RecalculateNormals();
            frameMeshObject.GetComponent<MeshFilter>().mesh = mesh;
            frameMeshObject.GetComponent<MeshCollider>().sharedMesh = mesh;
            //   mesh.UploadMeshData(false);
        }

        private void LoadMesh()
        {
            var meshInfo = new __MeshInfo();
            getMeshInfo(_cppContext, ref meshInfo);

            var vertexBuffer = new Vector3[meshInfo.m_vertex_count];
            var indexBuffer = new int[meshInfo.m_index_count];

            getMeshBuffers(ref meshInfo, vertexBuffer, indexBuffer);

            _meshDtoQueue.Enqueue(new __MeshDto
            {
                Triangles = indexBuffer,
                Vertices = vertexBuffer
            });
        }
    }
}