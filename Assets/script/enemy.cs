using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class enemy : MonoBehaviour {
    [SerializeField] GameObject explosion;
    [SerializeField] float health = 100;
    [SerializeField] float shotCounter;
    [SerializeField] float minTimeBetweenShots = 0.2f;
    [SerializeField] AudioClip aClip;
    [SerializeField] AudioClip bClip;
    [SerializeField] int enemyPoints = 5;
    [SerializeField] float LaserVolume = 10.0f;
    float maxTimeBetweenShots = 3f;
   public GameObject xProjectile;
    [SerializeField] float projectileSpeed=1f;
    private bool bombDropped=false;
    private bool goodHit = false;
    // Use this for initialization
    void Start () {
        shotCounter = Random.Range(minTimeBetweenShots, maxTimeBetweenShots);
     
        //Debug.Log("enemy laser from Start:"+xProjectile);
       
    }
	
	// Update is called once per frame
	void Update () {
        //Debug.Log("enemy laser from Update:" + xProjectile);
       
        CountDownAndShoot();
		
	}
    private void CountDownAndShoot()
    {
        shotCounter -= Time.deltaTime;
       if (shotCounter<=0f )
        {
            fire();
            bombDropped = true;
            shotCounter = Random.Range(minTimeBetweenShots, maxTimeBetweenShots);
        }
    }
    private void fire()
    {
        //Debug.Log();
        AudioSource.PlayClipAtPoint(aClip, Camera.main.transform.position, LaserVolume);
        Debug.Log("enemy firing!");
        GameObject plaser = Instantiate(xProjectile, transform.position,
                       Quaternion.identity) as GameObject;
        plaser.GetComponent<Rigidbody2D>().velocity =
                    new Vector2(0, -1f*projectileSpeed);
    }
    private void OnTriggerEnter2D(Collider2D other)
    {
        int HitByLaser = 0;
        Debug.Log("Collision:" + other);
        damageDealer damageDealer = other.gameObject.GetComponent<damageDealer>();
        if (other.name.Contains("Laser") )
            {
            HitByLaser = 1;
        }
        ProcessHit(damageDealer,HitByLaser);
    }

    private void ProcessHit(damageDealer damageDealer,int HitByLaser)
    {
        health -= damageDealer.GetDamage();
        damageDealer.Hit();
        if (health <= 00)
        {
            AudioSource.PlayClipAtPoint(bClip, Camera.main.transform.position);
            Destroy(gameObject);
            GameObject xExplode = Instantiate(explosion, transform.position,
                       Quaternion.identity) as GameObject;
            Destroy(xExplode,1f);
            if (HitByLaser == 1)
            {
                FindObjectOfType<keepScor>().addScore(enemyPoints);
            }
        }
    }
}
