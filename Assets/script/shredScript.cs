using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class shredScript : MonoBehaviour {
   
    private void OnTriggerEnter2D(Collider2D collision)
    {
        Debug.Log("Collision");
        Destroy(collision.gameObject);
    }
}
