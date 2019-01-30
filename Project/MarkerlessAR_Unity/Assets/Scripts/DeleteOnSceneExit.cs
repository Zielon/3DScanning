using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DeleteOnSceneExit : MonoBehaviour
{


    void Update()
    {
        if(transform.position.magnitude > 15.0f)
        {
            Destroy(gameObject);

        }
    }


    //Unity does some strange stuff when objects hit our mesh that triggers an interaction with the scene bounds
    //void OnTriggerEnter(Collider other)
    //{
    //    if (other.tag == "EndOfScene")
    //    {
    //        Debug.Log("object out of scene");
    //        Destroy(gameObject);

    //    }
    //}
}
