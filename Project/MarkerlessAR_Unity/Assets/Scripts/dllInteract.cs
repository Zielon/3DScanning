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
	Vector3 pos;
	Vector4 cubePos;
	Matrix4x4 camProj;
	Camera m_mainCam;
    System.IntPtr cppContext;
	Vector3 translation;
	Quaternion q;
	Vector3 scale;
	Matrix4x4 m; 


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
		pos = new Vector3 (0f, 0f, 0f);
		cubePos = new Vector4 (0f, 0f, 0f, 0f);

		//test the pose estimation transformation matrix
		translation = new Vector3(0.1554f, -1.1425f, 1.3593f); //initial position of camera
		q = new Quaternion ();
		q.Set (-0.5691f, 0.6454f, -0.3662f, 0.3541f);		 //initial orientation of camera
		scale = new Vector3 (1, 1, 1);
		m = Matrix4x4.TRS(translation, q, scale);			//Generate a 4x4 homogeneous transformation matrix from a 3D point, unit quaternion and scale vecotr.
		print ("homogenous trans matrix of camera in initial pose: \n" + m);

	
    }
	public static Quaternion QuaternionFromMatrix(Matrix4x4 m) {
		// Adapted from: http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
		Quaternion q = new Quaternion();
		q.w = Mathf.Sqrt( Mathf.Max( 0, 1 + m[0,0] + m[1,1] + m[2,2] ) ) / 2; 
		q.x = Mathf.Sqrt( Mathf.Max( 0, 1 + m[0,0] - m[1,1] - m[2,2] ) ) / 2; 
		q.y = Mathf.Sqrt( Mathf.Max( 0, 1 - m[0,0] + m[1,1] - m[2,2] ) ) / 2; 
		q.z = Mathf.Sqrt( Mathf.Max( 0, 1 - m[0,0] - m[1,1] + m[2,2] ) ) / 2; 
		q.x *= Mathf.Sign( q.x * ( m[2,1] - m[1,2] ) );
		q.y *= Mathf.Sign( q.y * ( m[0,2] - m[2,0] ) );
		q.z *= Mathf.Sign( q.z * ( m[1,0] - m[0,1] ) );
		return q;
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


		    //pos = GameObject.FindGameObjectWithTag("FlyingCube").transform.position; //get the cube position

			//cubePos.x = pos.x;
			//cubePos.y = pos.y;
			//cubePos.z = pos.z;
			//cubePos.w = 1f; // need 4 dim vector to apply tranformation pose on it.


			//parse the pose vector to get the columns
			Vector4 firstCol = new Vector4 (pose [0], pose [1], pose [2], pose [3]);
			Vector4 secCol = new Vector4 (pose [4], pose [5], pose [6], pose [7]);
			Vector4 thirdCol = new Vector4 (pose [8], pose [9], pose [10], pose [11]);
			Vector4 fourthCol = new Vector4 (pose [12], pose [13], pose [14], pose [15]);

			//Set the columns from pose to transformation matrix
			trans.SetColumn (0, firstCol);
			trans.SetColumn (1, secCol);
			trans.SetColumn (2, thirdCol);
			trans.SetColumn (3, fourthCol);

			Debug.Log ("pose estimation transformation matrix: \n" + trans.ToString("F5"));
			//Check if the pose estimation transformation matrix is working
			Matrix4x4 newPos = trans*m;
			Debug.Log("new pos and orientation of camera: \n " + newPos.ToString("F5"));
			Debug.Log("check the translation part: \n"+ newPos.GetColumn(3).ToString("F5"));
			Quaternion newQ = QuaternionFromMatrix (newPos);
			Debug.Log("transformed quaternion: \n" +  newQ.ToString("F5"));

			//Apply transformation
			//cubePos = trans*cubePos;

			//print("Cube after transformation "+  cubePos);
			//Apply transformation to the cube object
			//Camera intrinsics
			//cubePos.x = (cubePos.x*0.525f/1f)+ 0.319f;
			//cubePos.y = (cubePos.y * 0.525f / 1f) + 0.319f;
			//GameObject.FindGameObjectWithTag ("FlyingCube").transform.position = new Vector3 (cubePos.x, cubePos.y, 0);

            //Debug.Log("Sprite created successfuly");
        }
        else
        {
            Debug.Log("Could not read IMG");
        }
    }
}