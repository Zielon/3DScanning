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
    private static extern void trackerCameraPose(System.IntPtr context, byte[] image, float[] pose, int w, int h);

    [DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)]
    private static extern void dllMain(System.IntPtr context, byte[] image, float[] pose);


    [DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)] private static extern int getImageWidth(System.IntPtr context);

    [DllImport(DllFilePath, CallingConvention = CallingConvention.Cdecl)] private static extern int getImageHeight(System.IntPtr context);

    //shared memory
    byte[] image = null;
    float[] pose = null;
	Matrix4x4 trans;
	Vector3 pos = null;

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
		trans = Matrix4x4.identity;

    }

    // Update is called once per frame
    void Update()
    {
        Debug.Log("Update test");


		//Vector3 rot= GameObject.FindGameObjectWithTag("FlyingCube").transform.eulerAngles;
		//Debug.Log (pos);
		//Debug.Log (rot);
		//GameObject.FindGameObjectWithTag ("FlyingCube").transform.eulerAngles = new Vector3 (rot.x, rot.y +1, rot.z); 

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
		
		    pos = GameObject.FindGameObjectWithTag("FlyingCube").transform.position;
			//GameObject.FindGameObjectWithTag ("FlyingCube").transform.position = new Vector3 (pose[12], pose[13], pose[14]+100); 
			//parse the pose vector to get the columns
			Vector4 firstCol = new Vector4 (pose [0], pose [1], pose [2], pose [3]);
			Vector4 secCol = new Vector4 (pose [4], pose [5], pose [6], pose [7]);
			Vector4 thirdCol = new Vector4 (pose [8], pose [9], pose [10], pose [11]);
			Vector4 fourthCol = new Vector4 (pose [12], pose [13], pose [14], pose [15]);
			Vector4 cubePos;
			cubePos.x = pos.x;
			cubePos.y = pos.y;
			cubePos.z = pos.z;
			cubePos.w = 1f; // need 4 dim vector to apply tranformation pose on it.
			print("transformation matrix: \n" + trans);
			print("Cube before transformation \n" + cubePos);

			//Set the columns from pose to transformation matrix
			trans.SetColumn (0, firstCol);
			trans.SetColumn (1, secCol);
			trans.SetColumn (2, thirdCol);
			trans.SetColumn (3, fourthCol);

			//Apply transformation
			cubePos = trans*cubePos;
			print("Cube after transformation "+  cubePos);
			//Apply transformation to the cube object
			GameObject.FindGameObjectWithTag ("FlyingCube").transform.position = new Vector3 (cubePos.x, cubePos.y, cubePos.z);

            //Debug.Log("Sprite created successfuly");
        }
        else
        {
            Debug.Log("Could not read IMG");
        }
    }
}