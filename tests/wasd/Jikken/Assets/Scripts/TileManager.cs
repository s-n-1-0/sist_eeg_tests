using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TileManager : MonoBehaviour
{
    /// <summary>
    /// ‚±‚Ì”ÍˆÍ‚Í•K‚¸Šï”
    /// </summary>
    public int tileRange = 7;
    public int tileSize = 100;
    public GameObject tilePrefab;
    private RectTransform[][] tiles;
    void Start()
    {  

        void MakeTileRow(int i)
        {
            void MakeAndSetTile(int i, int j) => tiles[i + tileRange / 2][j + tileRange / 2] =  MakeTile(new Vector2(tileSize * j, tileSize * i));
            tiles[i + tileRange/2] = new RectTransform[tileRange];
            MakeAndSetTile(i, 0);
            for (int j = 1; j <= tileRange / 2; j++)
            {
                MakeAndSetTile(i, j);
                MakeAndSetTile(i, j * -1);
            }
        }
        tiles = new RectTransform[tileRange][];
        MakeTileRow(0);
        for (int i = 1; i <= tileRange/2; i++)
        {
            MakeTileRow(i);
            MakeTileRow(i * -1);
        }

    }
    RectTransform MakeTile(Vector2 pos)
    {
        var child = Instantiate(tilePrefab).GetComponent<RectTransform>();
        child.SetParent(transform);
        child.localPosition = pos;
        return child;
    }
    // Update is called once per frame
    void Update()
    {
        
    }
}
