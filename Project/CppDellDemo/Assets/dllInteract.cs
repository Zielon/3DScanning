using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading; 
using UnityEngine;

using UnityEngine.UI;
public class dllInteract : MonoBehaviour {

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
        /**/

    }
}
