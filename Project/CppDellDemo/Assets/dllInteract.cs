using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading; 
using System;
using UnityEngine;

using UnityEngine.UI;


public class dllInteract : MonoBehaviour
{
    private const string DllFilePath = "/Users/barisyazici/3DScanning/Project/Tracker/cmake-build-debug/lib/libtracker.dylib";

    //[DllImport ("ASimplePlugin")]
    //private static extern IntPtr PrintHello();
     
   [DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)] private static extern int test();
   [DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)] private static extern System.IntPtr createTracker();
   [DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)] private static extern int trackerCount(System.IntPtr tracker_object);

    [DllImport("tracker")]    private static extern IntPtr PrintHello();

    Thread cppMainThread;

    // Use this for initialization
    void Start()
    {
        cppMainThread = new Thread(cppMainThreadCaller);
        cppMainThread.Start(); 
    }


    void cppMainThreadCaller()
    {
        Debug.Log(Marshal.PtrToStringAuto (PrintHello()));
        /*
        Debug.Log("Test 4");
        int a = test();
        //int a = 5;

        Debug.Log( string.Format("My favorite number {0}\n", a) );

        System.IntPtr tracker = createTracker();

        Debug.Log("So far so good");

        int b = trackerCount(tracker);

        Debug.Log(string.Format("Class tracker count: {0}\n", b));

        Debug.Log("Final Test");
        */
    }

    // Update is called once per frame
    void Update()
    {

    }
}