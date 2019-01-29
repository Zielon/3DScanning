using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PaintBallScript : MonoBehaviour
{

    public GameObject SplatPrefab; 

    public float speed = .50f;

    public double deleteAfter = 5000;

    GameObject spawnedObject; 

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        transform.position += transform.forward * speed * Time.deltaTime;

        deleteAfter -= Time.deltaTime;

        if(deleteAfter<0)
        {
            Destroy(spawnedObject);
            Destroy(gameObject); 
        }

    }


    void OnCollisionEnter(Collision collision)
    {
        if(collision.gameObject.tag == "FrameMesh")
        {
            GameObject spawnedObject = Instantiate(SplatPrefab);
            spawnedObject.transform.position = GetComponent<Collider>().ClosestPoint(collision.contacts[0].point) + collision.contacts[0].normal * 0.001f ;
            spawnedObject.transform.forward = -collision.contacts[0].normal;      
        }

        GetComponent<Collider>().enabled = false;
        GetComponent<MeshRenderer>().enabled = false; 

    }

}
