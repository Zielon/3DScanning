using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading; 
using UnityEngine;

using UnityEngine.UI;


public class dllInteract : MonoBehaviour
{
    private const string DllFilePath = @"C:\Users\Lukas\Desktop\bin\tracker.dll";

    [DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)] private static extern int test();
    [DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)] private static extern System.IntPtr createTracker();
    [DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)] private static extern int trackerCount(System.IntPtr tracker_object);

    //[DllImport("tracker")] private static extern int test();

    Thread cppMainThread;

    // Use this for initialization
    void Start()
    {
        cppMainThread = new Thread(cppMainThreadCaller);
        cppMainThread.Start(); 
    }


    void cppMainThreadCaller()
    {
        Debug.Log("Test 4");
        int a = test();
        //int a = 5;

        Debug.Log( string.Format("My favorite number {0}\n", a) );

        System.IntPtr tracker = createTracker();

        Debug.Log("So far so good");

        int b = trackerCount(tracker);

        Debug.Log(string.Format("Class tracker count: {0}\n", b));

        Debug.Log("Final Test");
    }

    // Update is called once per frame
    void Update()
    {

    }
}

/*public class dllInteract : MonoBehaviour {

    RawImage rawImg;


    //[DllImport("UnityDLL")] private static extern int dllMain();
     

    [DllImport("UnityDLL")] private static extern System.IntPtr getImage(double time);

    Thread cppMainThread;

    // Use this for initialization
    void Start () {
        //cppMainThread = new Thread(cppMainThreadCaller);
       // cppMainThread.Start(); 
	}


    void cppMainThreadCaller()
    {
       // dllMain(); 
    }
	
	// Update is called once per frame
	void Update () {

        Debug.Log("Mark0"); 
        System.IntPtr cppImg = getImage(Time.time);
        Debug.Log("Mark1");
        byte[] result = new byte[640 * 480 * 3];

        Marshal.Copy(cppImg, result, 0, 640 * 480 * 3);

        Debug.Log("Mark2"); 
        Texture2D tex = new Texture2D(640, 480, TextureFormat.RGB24, false);
        tex.LoadRawTextureData(result);

        Debug.Log("Mark3"); 
        SpriteRenderer videoBg = GetComponent<SpriteRenderer>();
        videoBg.sprite = Sprite.Create(tex, new Rect(0, 0, tex.width, tex.height), new Vector2(.5f, .5f));
    }
}*/
