using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PaintBallScript : MonoBehaviour
{

    public GameObject SplatPrefab; 

    public float speed = .50f; 
  

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        transform.position += transform.forward * speed * Time.deltaTime; 
    }


    void OnCollisionEnter(Collision collision)
    {
        if(collision.gameObject.tag == "FrameMesh")
        {
            GameObject splat = Instantiate(SplatPrefab);
            splat.transform.position = GetComponent<Collider>().ClosestPoint(collision.contacts[0].point) + collision.contacts[0].normal;
            splat.transform.forward = -collision.contacts[0].normal;      
        }

        Destroy(gameObject); 
    }

}
