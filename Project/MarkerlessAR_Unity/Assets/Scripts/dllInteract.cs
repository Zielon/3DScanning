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

    //[DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)] private static extern int test();

    [DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)] private static extern System.IntPtr createContext();
    [DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)]
    private static extern void trackerCameraPose(System.IntPtr context,
     byte[] image, float[] pose, int w, int h);

    [DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)]
    private static extern void dllMain(System.IntPtr context, byte[] image, float[] pose);


    [DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)] private static extern int getImageWidth(System.IntPtr context);

    [DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)] private static extern int getImageHeight(System.IntPtr context);

    //shared memory
    byte[] image = null;
    float[] pose = null;

    System.IntPtr cppContext;


    //general setup
    int w = -1;
    int h = -1;

    // Use this for initialization
    void Start()
    {
        Debug.Log("Creating Context");
        
        //Debug.Log(test());

        cppContext = createContext();

        w = getImageWidth(cppContext);
        h = getImageHeight(cppContext);

        Debug.Log("Created Contex. Image dimensions: " + w + "x" + h);

        pose = new float[16];
        image = new byte[w * h * 3];
    }

    // Update is called once per frame
    void Update()
    {
        Debug.Log("Update test");

        dllMain(cppContext, image, pose);


        if (image != null)
        {
            //Create texture from image
            Texture2D tex = new Texture2D(w, h, TextureFormat.RGB24, false);

            tex.LoadRawTextureData(image);
            tex.Apply(); 

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