using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraSettings : MonoBehaviour
{
    public enum MeshRenderMode { DISABLED, TRANSPARENT, OPAQUE };

    public MeshRenderMode meshRenderMode = MeshRenderMode.TRANSPARENT; 
}
