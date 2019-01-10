using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DeleteOnSceneExit : MonoBehaviour
{
    void OnTriggerEnter(Collider other)
    {
        if (other.tag == "EndOfScene")
        {
            Destroy(gameObject);
        }
    }
}
