using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class zPrint : MonoBehaviour {
    
	// Use this for initialization
	void Start () {
        Debug.Log("hi matt");
        StartCoroutine("sqPrint");
     
	}
	
	// Update is called once per frame
	void Update () {
		
	}
    
    IEnumerator sqPrint()
    {
        while (true)
        {
            Debug.Log("hello");
            yield return new WaitForSeconds(2f);
        }
    }
}
