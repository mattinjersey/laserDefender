using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class enemySpawner : MonoBehaviour {

    [SerializeField] List<waveConfig> waveConfigs;
    [SerializeField] int startingWave = 0;
    [SerializeField] bool looping = false;
    int numEnemies = 5;
    IEnumerator Start() {
        do { 
        yield return StartCoroutine(SpawnAllWaves());
    }
        while (looping);

	}
	private IEnumerator SpawnAllWaves()
    {
        for  (int  waveIndex= startingWave;waveIndex<waveConfigs.Count;waveIndex++)
        {
            var currentWave = waveConfigs[waveIndex];
            yield return StartCoroutine(SpawnAllEnemiesInWave(currentWave));
        }
    }
	private IEnumerator SpawnAllEnemiesInWave(waveConfig waveConfig)
    {
        for (int i = 0; i < waveConfig.GetnumberOfEnemies(); i++)
        {

            var newEnemy=Instantiate(
                waveConfig.GetEnemyPrefab(),
                waveConfig.GetWaypoints()[0].transform.position,
                Quaternion.identity);
            newEnemy.GetComponent<EnemyPath>().SetWaveConfig(waveConfig);
            yield return new WaitForSeconds(waveConfig.GettimeBetweenSpawns());
        }

    }
}
