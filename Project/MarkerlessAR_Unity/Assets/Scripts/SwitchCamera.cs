using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SwitchCamera : MonoBehaviour
{
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
    }

    public void cycleNextCam()
    {
        if (cameras.Length == 0) return;

        cameras[currentCamera].enabled = false;
        currentCamera = (currentCamera + 1) % cameras.Length;
        cameras[currentCamera].enabled = true;

    }

}
