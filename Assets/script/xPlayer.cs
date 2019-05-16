using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
public class xPlayer : MonoBehaviour {
   [SerializeField] float aSpeed=10f;
    [SerializeField] float health = 100f;
    [SerializeField] float padding = 1f;
    [SerializeField] GameObject aLaser;
    [SerializeField] float projectileSpeed=10f;
    [SerializeField] float projectileFiringPeriod = 1f;
    [SerializeField] AudioClip aClip;
    [SerializeField] AudioClip bClip;
    [SerializeField] float aVolume = 10.0f;

    float xMin,xMax,yMin,yMax;
    Coroutine aFire;
	// Use this for initialization
	void Start () {
        SetUpMoveBoundaries();
        Debug.Log("player laser" + aLaser);
	}

    private void SetUpMoveBoundaries()
    {
        Camera gameCamera=Camera.main;
        xMin = gameCamera.ViewportToWorldPoint(new Vector3(0, 0, 0)).x;
        xMax = gameCamera.ViewportToWorldPoint(new Vector3(1, 0, 0)).x;
        yMin = gameCamera.ViewportToWorldPoint(new Vector3(0, 0, 0)).y;
        yMax = gameCamera.ViewportToWorldPoint(new Vector3(0, 1, 0)).y;
    }

    // Update is called once per frame
    void Update () {
        Move();
        Fire();
	}

    private void Fire()
    {

        if (Input.GetButtonDown("Fire1"))
        {
            aFire=StartCoroutine("sqPrint");
        }
         if (Input.GetButtonUp("Fire1"))
        {
            StopCoroutine(aFire);
        }
    }
    IEnumerator sqPrint()
    {
        while (true)
        {
            AudioSource.PlayClipAtPoint(aClip, Camera.main.transform.position, aVolume);
            GameObject laser = Instantiate(aLaser, transform.position,
                   Quaternion.identity) as GameObject;
            laser.GetComponent<Rigidbody2D>().velocity =
                new Vector2(0, projectileSpeed);
            yield return new WaitForSeconds(projectileFiringPeriod);
        }
    }
        private void OnTriggerEnter2D(Collider2D other)
        {
            Debug.Log("Collision:" + other);
            damageDealer damageDealer = other.gameObject.GetComponent<damageDealer>();
            ProcessHit(damageDealer);
        }
    public float GetHealth()
    {
        return health;
    }
        private void ProcessHit(damageDealer damageDealer)
        {
        AudioSource.PlayClipAtPoint(bClip, Camera.main.transform.position,aVolume);
        health -= damageDealer.GetDamage();
            damageDealer.Hit();
            if (health <= 00)
            {
                Destroy(gameObject);
            Debug.Log("destroy player!");
            FindObjectOfType<gameManage>().playerShot();
         
            }
        }







    private void Move()
    {
        var deltaX = Input.GetAxis("Horizontal")*Time.deltaTime*aSpeed;
        var deltaY = Input.GetAxis("Vertical") * Time.deltaTime * aSpeed;
        var newXPos = transform.position.x + deltaX;
        var newYPos = transform.position.y + deltaY;
        newXPos = Mathf.Clamp(newXPos, xMin+padding, xMax-padding);
        newYPos = Mathf.Clamp(newYPos, yMin+padding, yMax-padding);
        
        transform.position = new Vector2(newXPos, newYPos);

    }
}
