using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class aRotateBomb : MonoBehaviour {
    [SerializeField] float rotationSpeed = 0.1f;
   Vector3 rotationVec=new Vector3(0.0f,0.0f,1.0f);
	// Use this for initialization

	
	// Update is called once per frame
	void Update () {
        gameObject.transform.Rotate(rotationVec*Time.deltaTime*rotationSpeed);
	}
}
