using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class text : MonoBehaviour {
    keepScor keepScor;
    int thisScore;
    Text Stext;
	// Use this for initialization
	void Start () {
        keepScor = FindObjectOfType<keepScor>();
        Stext = GetComponent<Text>();
	}
	
	// Update is called once per frame
	void Update () {
        thisScore = keepScor.getScore();
        Stext.text = thisScore.ToString();
	}
}
