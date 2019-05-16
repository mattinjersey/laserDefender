using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
public class gameManage : MonoBehaviour {

    [SerializeField] int deadPeriod = 1;
    // Use this for initialization
    void Start () {
        Debug.Log("big cooky"); 
	}
	
	// Update is called once per frame
	void Update () {
		
	}
    public void buttonClicked()
    {
        Debug.Log("button clicked!");
        SceneManager.LoadScene("game");
    }
    public void playerShot()
    {
        Debug.Log("player shot!");
        StartCoroutine("youDead");
        
    }

    IEnumerator youDead()
    {
       
            Debug.Log("player got hit!");
            yield return new WaitForSeconds(deadPeriod);
        SceneManager.LoadScene("gameOver");
    }
}
