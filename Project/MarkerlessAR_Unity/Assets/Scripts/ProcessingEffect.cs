using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI; 

public class ProcessingEffect : MonoBehaviour
{

    public Text text;

    float update = -0.75f; 

    // Update is called once per frame
    void Update()
    {
        Color c = text.color;
        c.a += update * Time.deltaTime;

        if(c.a<=0)
        {
            c.a = 0;
            update = update<0? -update : update; 
        }else if(c.a>=1)
        {
            c.a = 1;
            update = update < 0 ? update : -update;
        }
        text.color = c;
    }
}
