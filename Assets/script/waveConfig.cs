using System.Collections;
using System.Collections.Generic;
using UnityEngine;
[CreateAssetMenu(menuName ="Enemy Wave Config")]
public class waveConfig : ScriptableObject {
   
    [SerializeField] GameObject enemyPrefab;
    [SerializeField] GameObject pathPreFab;
    [SerializeField] float timeBetweenSpawns = 0.5f;
    [SerializeField] float spawnRandomFactor = 0.3f;
    [SerializeField] int numberOfEnemies = 5;
    [SerializeField] float moveSpeed = 2f;
public GameObject GetEnemyPrefab()
    { return enemyPrefab;   }

    public List<Transform> GetWaypoints()
    {
        var waveWaypoints = new List<Transform>();
        foreach (Transform child in pathPreFab.transform)
        {
            waveWaypoints.Add(child);
        }
        return waveWaypoints;
    }
    public float GettimeBetweenSpawns()
    { return timeBetweenSpawns; }
    public float GetspawnRandomFactor()
    { return spawnRandomFactor; }
    public int GetnumberOfEnemies()
    { return numberOfEnemies; }
    public float GetmoveSpeed()
    { return moveSpeed; }
}
