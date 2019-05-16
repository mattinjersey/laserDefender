using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
public class keepScor : MonoBehaviour {
   
    [SerializeField] int gameScore = 0;

    private void Awake()
    {
        SetUpSingleton();

    }

    private void SetUpSingleton()
    {
        int numGameSessions = FindObjectsOfType<keepScor>().Length;
        if (numGameSessions>1)
        {
            Destroy(gameObject);
        }
        else
        {
            DontDestroyOnLoad(gameObject);
        }
    }
    void Start () {
        Debug.Log(gameScore);
	}
	
	// Update is called once per frame
	void Update () {
       
    }
    public void ResetGame()
    {
        Destroy(gameObject);
    }
    public void addScore(int aScore)
    {
        gameScore += aScore;
     

        Debug.Log(gameScore);
    }
    public int getScore()
    {
        return gameScore;
    }
}
