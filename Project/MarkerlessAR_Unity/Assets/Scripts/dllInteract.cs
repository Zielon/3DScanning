using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading; 
using UnityEngine;

using UnityEngine.UI;


public class dllInteract : MonoBehaviour
{
    //Unity automatically find DLL files located on Assets/Plugins
    private const string DllFilePath = @"Tracker";
    
    [DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)] private static extern System.IntPtr createTracker();
    [DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)] private static extern void trackerCameraPose(System.IntPtr tracker_object,
     byte[] image, float [] pose, int w, int h);

    //General setup image
    byte[] image = null;
    int w = 640;
    int h = 480;

    Thread cppMainThread;

    // Use this for initialization
    void Start()
    {
        cppMainThread = new Thread(cppMainThreadCaller);
        cppMainThread.Start(); 
    }


    void cppMainThreadCaller()
    {
        
        //Debug.Log("Thread Start");

        System.IntPtr tracker = createTracker();

        w = 640;
        h = 480;
        float[] pose = new float[16];
        image = new byte[w * h * 3];

        trackerCameraPose(tracker, image, pose, w, h);
    }

    // Update is called once per frame
    void Update()
    {
        //Debug.Log("Update test");

        if (image != null)
        {
            //Create texture from image
            Texture2D tex = new Texture2D(w, h, TextureFormat.RGB24, false);
            tex.LoadRawTextureData(image);

            //Debug.Log("Texture created successfuly");

            Image videoBg = GetComponent<Image>();
            videoBg.sprite = Sprite.Create(tex, new Rect(0, 0, tex.width, tex.height), new Vector2(.5f, .5f));

            //Debug.Log("Sprite created successfuly");
        }
        else
        {
            Debug.Log("Could not read IMG"); 
        }
    }
}