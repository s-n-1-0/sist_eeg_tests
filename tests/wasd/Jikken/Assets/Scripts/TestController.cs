using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TestController : MonoBehaviour
{
    public RoadGenerator road;
    public TileManager tiles;
    void Start()
    {
        Draw();
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.W))
        {
            
            road.MakeNextRoad();
            Draw();
        }
    }
    public void Draw()
    {
        tiles.ResetTileColor();
        var nowPos = new Vector2(tiles.tileRange / 2, tiles.tileRange / 2);
        tiles.tiles[tiles.tileRange / 2][tiles.tileRange / 2].color = Color.red;
        for (int i = 1; i < 10; i++)
        {
            if (-1 == road.nowHistoryIndex - i) break;
            var d = road.history[road.nowHistoryIndex - i];
            nowPos += d.GetDirectionVec2() * -1;
            if (nowPos.x >= tiles.tileRange || nowPos.x < 0 || nowPos.y >= tiles.tileRange || nowPos.y < 0) continue;
            tiles.tiles[(int)nowPos.y][(int)nowPos.x].ChangeColor(true);
        }
        nowPos = new Vector2(tiles.tileRange / 2, tiles.tileRange / 2);
        for (int i = 0; i < 10; i++)
        {
            if (road.history.Count == road.nowHistoryIndex + i) break;
            var d = road.history[road.nowHistoryIndex + i];
            nowPos += d.GetDirectionVec2();
            if (nowPos.x >= tiles.tileRange || nowPos.x < 0 || nowPos.y >= tiles.tileRange || nowPos.y < 0) continue;
            tiles.tiles[(int)nowPos.y][(int)nowPos.x].ChangeColor(true);
        }
    }
}
