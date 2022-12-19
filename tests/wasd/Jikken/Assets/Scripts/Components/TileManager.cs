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
    public TileImage[][] tiles;
    void Awake()
    {  

        void MakeTileRow(int i)
        {
            void MakeAndSetTile(int i, int j) => tiles[i + tileRange / 2][j + tileRange / 2] =  MakeTile(new Vector2(tileSize * j, tileSize * i));
            tiles[i + tileRange/2] = new TileImage[tileRange];
            MakeAndSetTile(i, 0);
            for (int j = 1; j <= tileRange / 2; j++)
            {
                MakeAndSetTile(i, j);
                MakeAndSetTile(i, j * -1);
            }
        }
        tiles = new TileImage[tileRange][];
        MakeTileRow(0);
        for (int i = 1; i <= tileRange/2; i++)
        {
            MakeTileRow(i);
            MakeTileRow(i * -1);
        }
    }
    TileImage MakeTile(Vector2 pos)
    {
        var child = Instantiate(tilePrefab).GetComponent<RectTransform>();
        child.SetParent(transform);
        child.localPosition = pos;
        return child.GetComponent<TileImage>();
    }
    public void ResetTileColor()
    {
        for (int i = 0; i < tileRange; i++) for (int j = 0; j < tileRange; j++) tiles[i][j].ChangeColor(null);
    }
}
