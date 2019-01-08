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
        private const string DllFilePath = @"Tracker_release";

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
            }
            else
            {
                Debug.Log("Could not read IMG");
            }
        }
    }
}