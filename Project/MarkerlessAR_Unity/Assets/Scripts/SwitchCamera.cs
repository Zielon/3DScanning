using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SwitchCamera : MonoBehaviour
{

    public Material frameMeshTransparent;
    public Material frameMeshOpaque;
    public GameObject frameMeshObject;

    public Camera[] cameras;
    private int currentCamera = 0; 

    // Start is called before the first frame update
    void Start()
    {
        if (cameras.Length == 0) return;

        for (int i =1; i<cameras.Length; ++i)
        {
            cameras[i].enabled = false; 
        }

        applyCameraSettings(cameras[currentCamera].GetComponent<CameraSettings>()); 
    }

    public void cycleNextCam()
    {
        if (cameras.Length == 0) return;

        cameras[currentCamera].enabled = false;
        currentCamera = (currentCamera + 1) % cameras.Length;
        cameras[currentCamera].enabled = true;


        CameraSettings settings = cameras[currentCamera].gameObject.GetComponent<CameraSettings>();
        if (settings == null)
        {
            Debug.Log("Could not find camera settings"); 
            settings = new CameraSettings(); 
        }
        applyCameraSettings(settings);

    }

    private void applyCameraSettings(CameraSettings settings)
    {
        frameMeshObject.GetComponent<MeshRenderer>().enabled = true;

        switch (settings.meshRenderMode)
        {
            case CameraSettings.MeshRenderMode.DISABLED:
            {
                frameMeshObject.GetComponent<MeshRenderer>().enabled = false; 
                break; 
            }
            case CameraSettings.MeshRenderMode.TRANSPARENT:
            {
                 frameMeshObject.GetComponent<MeshRenderer>().material = frameMeshTransparent;
                break;
            }
            case CameraSettings.MeshRenderMode.OPAQUE:
            {
                 frameMeshObject.GetComponent<MeshRenderer>().material = frameMeshOpaque;
                 break;
            }
        }
    }
}
