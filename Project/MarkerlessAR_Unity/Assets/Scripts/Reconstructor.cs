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

        private IntPtr _cppContext;
        private int _h = -1;

        //shared memory
        private byte[] _image;
        private float[] _pose;

        //general setup
        private int _w = -1;

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

        string trajectoryPath = string.Empty;
        Matrix4x4 poseTrans = new Matrix4x4();
        Vector3 translation ;
        Quaternion q;
        Vector3 scale;
        Matrix4x4 initialCamPose ;


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

            //Get the trajectory file
            trajectoryPath = absolutePath + "\\groundtruth.txt";
          
            //Get the position and orientation of the camera
            string[] lines = System.IO.File.ReadAllLines(trajectoryPath);
            StringReader strReader = new StringReader(lines[3]);        //First line after comments
            string[] words = strReader.ReadToEnd().Split(' ');
            var map = new Dictionary<string, float>();
            string[] orient = new string[] { "tx", "ty", "tz", "qx", "qy", "qz", "qw" };

            
            for (int i = 0; i < words.Length; i++) //Get float from string. Add it to map var.
            {
                if (i > 0)
                    map.Add(orient[i - 1], float.Parse(words[i], System.Globalization.CultureInfo.InvariantCulture));
         
            }
            

            //test the pose estimation transformation matrix
            translation = new Vector3(map["tx"], map["ty"], map["tz"]); //initial position of camera
            q = new Quaternion();
            q.Set(map["qx"], map["qy"], map["qz"], map["qw"]);         //initial orientation of camera
            scale = new Vector3(1, 1, 1);
            initialCamPose = Matrix4x4.TRS(translation, q, scale);           //Generate a 4x4 homogeneous transformation matrix from a 3D point, unit quaternion and scale vecotr.


            _w = getImageWidth(_cppContext);
            _h = getImageHeight(_cppContext);

            Debug.Log("Created Contex. Image dimensions: " + _w + "x" + _h);

            _pose = new float[16];
            _image = new byte[_w * _h * 3];
        }

        public static Quaternion QuaternionFromMatrix(Matrix4x4 m)
        {
            // Adapted from: http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
            Quaternion q = new Quaternion();
            q.w = Mathf.Sqrt(Mathf.Max(0, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2;
            q.x = Mathf.Sqrt(Mathf.Max(0, 1 + m[0, 0] - m[1, 1] - m[2, 2])) / 2;
            q.y = Mathf.Sqrt(Mathf.Max(0, 1 - m[0, 0] + m[1, 1] - m[2, 2])) / 2;
            q.z = Mathf.Sqrt(Mathf.Max(0, 1 - m[0, 0] - m[1, 1] + m[2, 2])) / 2;
            q.x *= Mathf.Sign(q.x * (m[2, 1] - m[1, 2]));
            q.y *= Mathf.Sign(q.y * (m[0, 2] - m[2, 0]));
            q.z *= Mathf.Sign(q.z * (m[1, 0] - m[0, 1]));
            return q;
        }


        // Update is called once per frame
        private void Update()
        {
            Debug.Log("Update test");

            dllMain(_cppContext, _image, _pose);


            if (_image != null)
            {
                //Create texture from image
                var tex = new Texture2D(_w, _h, TextureFormat.RGB24, false);

                tex.LoadRawTextureData(_image);
                tex.Apply();

                //Debug.Log("Texture created successfuly");

                var videoBg = GetComponent<Image>();
                videoBg.sprite = Sprite.Create(tex, new Rect(0, 0, tex.width, tex.height), new Vector2(.5f, .5f));

                //Debug.Log("Sprite created successfuly");

                // Get the transformation matrix between frames
                Vector4 firstCol = new Vector4(_pose[0], _pose[1], _pose[2], _pose[3]);      //parse the pose vector to get the columns
                Vector4 secCol = new Vector4(_pose[4], _pose[5], _pose[6], _pose[7]);
                Vector4 thirdCol = new Vector4(_pose[8], _pose[9], _pose[10], _pose[11]);
                Vector4 fourthCol = new Vector4(_pose[12], _pose[13], _pose[14], _pose[15]);

                //Set the columns from pose to transformation matrix
                poseTrans.SetColumn(0, firstCol);
                poseTrans.SetColumn(1, secCol);
                poseTrans.SetColumn(2, thirdCol);
                poseTrans.SetColumn(3, fourthCol);

                Debug.Log("pose estimation transformation matrix: \n" + poseTrans.ToString("F5"));
                //Check if the pose estimation transformation matrix is correct
                Matrix4x4 newPos = initialCamPose * poseTrans;
                Debug.Log("new pos and orientation of camera: \n " + newPos.ToString("F5"));
                Debug.Log("check the translation part: \n" + newPos.GetColumn(3).ToString("F5"));
                Quaternion newQ = QuaternionFromMatrix(newPos);
                Debug.Log("transformed quaternion: \n" + newQ.ToString("F5"));
            }
            else
            {
                Debug.Log("Could not read IMG");
            }
        }
    }
}