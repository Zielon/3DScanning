using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using UnityEngine;
using UnityEngine.UI;

namespace Assets.Scripts
{
    public class Reconstructor : MonoBehaviour
    {


        //Unity automatically find DLL files located on Assets/Plugins
        private const string DllFilePath = @"Tracker";

        //Unity vars

        public GameObject cameraRig;
        public GameObject frameMeshPrefab;


        private IntPtr _cppContext;

        //shared memory
        private byte[] _image;
        private float[] _pose;

        //general setup
        private int _w = -1;
        private int _h = -1;


        private LinkedList<Mesh> frameMeshes = new LinkedList<Mesh>(); 


        //[DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)] private static extern int test();

        [DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr createContext(byte[] path);

        [DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)]
        private static extern void trackerCameraPose(IntPtr context,
            byte[] image, float[] pose, int w, int h);

        [DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)]
        private static extern void dllMain(IntPtr context, byte[] image, float[] pose);

        [DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)]
        private static extern int getImageWidth(IntPtr context);

        [DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)]
        private static extern int getImageHeight(IntPtr context);

        [DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)]
        private static extern int getVertexCount(IntPtr context);

        [DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)]
        private static extern void getVertexBuffer(IntPtr context, [In,Out]Vector3[] vertexBuffer);

        [DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)]
        private static extern int getIndexCount(IntPtr context);

        [DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)]
        private static extern void getIndexBuffer(IntPtr context, [In,Out]int[] indexBuffer);

        // Use this for initialization
        private void Start()
        {
            Debug.Log("Creating Context");

            var segments = new List<string>(
                    Application.dataPath.Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar))
                {"..", "Datasets", "freiburg", " "};

            var absolutePath = segments.Aggregate(
                (path, segment) => path += Path.AltDirectorySeparatorChar + segment).Trim();

            _cppContext = createContext(Encoding.ASCII.GetBytes(absolutePath));

            _w = getImageWidth(_cppContext);
            _h = getImageHeight(_cppContext);

            Debug.Log("Created Contex. Image dimensions: " + _w + "x" + _h);

            _pose = new float[16];
            _image = new byte[_w * _h * 3];
        }

        // Update is called once per frame
        private void Update()
        {
         //   Debug.Log("Update test");

            dllMain(_cppContext, _image, _pose);

            //Create texture from image
            var tex = new Texture2D(_w, _h, TextureFormat.RGB24, false);

            tex.LoadRawTextureData(_image);
            tex.Apply();

            //Debug.Log("Texture created successfuly");

            var videoBg = GetComponent<Image>();
            videoBg.sprite = Sprite.Create(tex, new Rect(0, 0, tex.width, tex.height), new Vector2(.5f, .5f));

            //Debug.Log("Sprite created successfuly");

            // Apply camera poses
            Vector4 firstCol = new Vector4(_pose[0], _pose[1], _pose[2], _pose[3]);
            Vector4 secCol = new Vector4(_pose[4], _pose[5], _pose[6], _pose[7]);
            Vector4 thirdCol = new Vector4(_pose[8], _pose[9], _pose[10], _pose[11]);
            Vector4 fourthCol = new Vector4(_pose[12], _pose[13], _pose[14], _pose[15]);
            Matrix4x4 pose = new Matrix4x4();


            //Set the columns from pose to transformation matrix
            pose.SetColumn(0, firstCol);
            pose.SetColumn(1, secCol);
            pose.SetColumn(2, thirdCol);
            pose.SetColumn(3, fourthCol);

          //  Debug.Log("transformation matrix: \n" + pose);

            cameraRig.transform.position = fourthCol * 1000;
            cameraRig.transform.rotation = pose.rotation;

         //   Debug.Log("Pos: " + cameraRig.transform.position);
         //   Debug.Log("Rot: " + cameraRig.transform.rotation.eulerAngles);


            spawnFrameMesh(); 


        }


        private void spawnFrameMesh()
        {
            int vertexCount = getVertexCount(_cppContext);
            int indexCount = getIndexCount(_cppContext);

            Vector3[] vertexBuffer = new Vector3[vertexCount];
            int[] indexBuffer = new int[indexCount];
            getVertexBuffer(_cppContext, vertexBuffer);
            getIndexBuffer(_cppContext, indexBuffer);



            Mesh mesh = new Mesh();
            mesh.name = "Frame Mesh " + Time.frameCount; 
            mesh.vertices = vertexBuffer;
            mesh.triangles = indexBuffer; 
            mesh.RecalculateBounds();
            GameObject frameMeshObject = Instantiate(frameMeshPrefab);
            frameMeshPrefab.GetComponent<MeshFilter>().mesh = mesh;
          //  frameMeshObject.GetComponent<MeshCollider>().sharedMesh = mesh;

            frameMeshes.AddLast(mesh);


            Debug.Log("Loaded mesh with " + mesh.vertexCount + " verts and " + mesh.triangles.Length + " indices."); 

        }

    }
}


/*
 

            Vector3[] dummyvertexBuffer = {
                // front
             new Vector3(   -1.0f, -1.0f,  1.0f),
           new Vector3(   1.0f, -1.0f,  1.0f),
           new Vector3(   1.0f,  1.0f,  1.0f),
           new Vector3(  -1.0f,  1.0f,  1.0f),
            // back
         new  Vector3(   -1.0f, -1.0f, -1.0f),
         new  Vector3(    1.0f, -1.0f, -1.0f),
         new  Vector3(    1.0f,  1.0f, -1.0f),
          new  Vector3(  -1.0f,  1.0f, -1.0f)
          };

            vertexBuffer = dummyvertexBuffer;

            int[] dummIndexBuffer = {
		        // front
		        0, 1, 2,
                2, 3, 0,
		        // right
		        1, 5, 6,
                6, 2, 1,
		        // back
		        7, 6, 5,
                5, 4, 7,
		        // left
		        4, 0, 3,
                3, 7, 4,
		        // bottom
		        4, 5, 1,
                1, 0, 4,
		        // top
		        3, 2, 6,
                6, 7, 3
            };

            indexBuffer = dummIndexBuffer; 
     
     
     */
