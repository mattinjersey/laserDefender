using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class qText : MonoBehaviour {
    Text Stext;
    xPlayer aPlay;
	// Use this for initialization
	void Start () {
        Stext = GetComponent<Text>();
        aPlay = FindObjectOfType<xPlayer>();
	}
	
	// Update is called once per frame
	void Update () {
        float gameHealth;
        gameHealth = aPlay.GetHealth();
        Stext.text = gameHealth.ToString();
	}
}
