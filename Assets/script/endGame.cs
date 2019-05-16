using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
public class endGame : MonoBehaviour {

	// Use this for initialization
	void Start () {
		
	}
	public void switchScenes(int inputVal)
    {
        if (inputVal == 1)
        {
            Application.Quit();
        }
                else
        {
            int numGameSessions = FindObjectsOfType<keepScor>().Length;
            if (numGameSessions > 0)
            {
                GameObject.Find("gameScore").GetComponent<keepScor>().ResetGame();

            }
            SceneManager.LoadScene("startMenu");
        }

    }
	// Update is called once per frame
	void Update () {
		
	}
}
