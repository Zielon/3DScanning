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
  

    public class OfflineReconstructor : MonoBehaviour
    {
        public enum ProcessingState { INITIAL, RECORDING,LOAD_MESH, INTERACT };
        public ProcessingState currentState = ProcessingState.INITIAL; 




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
        public bool _use_sensor = false;
        private int _w = -1;

        public int abortAfterNFrames = -1;
        //bool _use_reconstruction = true;
        //bool _use_fusion = true;

        // Unity injected vars
        public GameObject cameraRig;
        public GameObject interactUI; 
        public GameObject frameMeshObject;
        public GameObject processingText; 
        public int meshUpdateRate = 15;
        public Image videoBG;

        [DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr createContext(byte[] path);

        [DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr createSensorContext(byte[] path);

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
        private static extern void enableReconstruction(IntPtr context, bool enable);

        [DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)]
        private static extern void getFrame(IntPtr context, byte[] image, bool record);

        [DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)]
        private static extern void computeOfflineReconstruction(IntPtr context, ref __MeshInfo mesh, float[] pose);

        [DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)]
        private static extern void deleteContext(IntPtr context); 

        
        // Use this for initialization
        private void Start()
        {
            var segments = new List<string>(
        Application.dataPath.Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar))
                    {"..", "Datasets", "freiburg", " "};

            var absolutePath = segments.Aggregate(
                (path, segment) => path += Path.AltDirectorySeparatorChar + segment).Trim();
            if (_use_sensor)
            {
                _cppContext = createSensorContext(Encoding.ASCII.GetBytes(absolutePath));
            }
            else
            {


                _cppContext = createContext(Encoding.ASCII.GetBytes(absolutePath));
            }

            _w = getImageWidth(_cppContext);
            _h = getImageHeight(_cppContext);

            Debug.Log("Created Context. Image dimensions: " + _w + "x" + _h);
            enableReconstruction(_cppContext, false); 

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
            _framesProcessed++;


            switch(currentState)
            {
                case ProcessingState.INITIAL:
                    {
                        getFrame(_cppContext, _image, false); 
                        break; 
                    }
                case ProcessingState.RECORDING:
                    {
                        getFrame(_cppContext, _image, true);
                        break;
                    }
                case ProcessingState.LOAD_MESH:
                    {
                        if (_meshDtoQueue.Count > 0)
                            AddMesh(_meshDtoQueue.Dequeue());
                        return; 
                    }
                case ProcessingState.INTERACT:
                    {
                        tracker(_cppContext, _image, _pose);
                        var pose = Helpers.GetPose(_pose);

                        cameraRig.transform.position = pose.GetColumn(3);
                        cameraRig.transform.rotation = pose.rotation;
                        break;
                    }
            }

            //Create texture from image
            var tex = new Texture2D(_w, _h, TextureFormat.RGB24, false);

            tex.LoadRawTextureData(_image);
            tex.Apply();

            videoBG.sprite = Sprite.Create(tex, new Rect(0, 0, tex.width, tex.height), new Vector2(.5f, .5f));


        }

        private void AddMesh(MeshDto dto)
        {
            var mesh = new Mesh {vertices = dto.Vertices, triangles = dto.Triangles};

            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            frameMeshObject.GetComponent<MeshFilter>().mesh = mesh;
            frameMeshObject.GetComponent<MeshCollider>().sharedMesh = mesh;

            processingText.SetActive(false);
            var pose = Helpers.GetPose(_pose);

            cameraRig.transform.position = pose.GetColumn(3);
            cameraRig.transform.rotation = pose.rotation;
            currentState = ProcessingState.INTERACT;
            interactUI.SetActive(true); 
        }

        private Thread SpawnFrameMeshThread()
        {
            return new Thread(() =>
            {
                var meshInfo = new __MeshInfo();
                computeOfflineReconstruction(_cppContext, ref meshInfo, _pose); 

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


        public void startRecording()
        {
            Debug.Log("Start Recording");
            Mesh empty = new Mesh(); 
            frameMeshObject.GetComponent<MeshFilter>().mesh = empty;
            frameMeshObject.GetComponent<MeshCollider>().sharedMesh = empty;
            currentState = ProcessingState.RECORDING;
        }

        public void stopRecording()
        {
            Debug.Log("Stop Recording");
            currentState = ProcessingState.LOAD_MESH;
            _thread = SpawnFrameMeshThread();
            _thread.Start();
            processingText.SetActive(true); 

        }


        void OnApplicationQuit()
        {
            deleteContext(_cppContext); 
            Debug.Log("Application ending after " + Time.time + " seconds");
        }

    }
}