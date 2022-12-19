using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TestController : MonoBehaviour
{
    public RoadGenerator road;
    public TileManager tiles;
    private CsvExporter csv;
    public Color beforeColor,afterColor,waitColor = Color.red,goColor = Color.green;
    public float waitTime = 1f;
    private bool isClicked = false;
    void Start()
    {
        csv = new CsvExporter();
        Draw();
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Alpha9))
        {
            Debug.Log("“¯Šú");
            csv.Record("sync","");
        }
        if (Input.GetKeyDown(KeyCode.Alpha0))
        {
            Debug.Log("Œv‘ªI—¹");
            csv.Close();
        }
        if (Input.GetKeyDown(KeyCode.W))
        {
            Move(RoadDirection.Up);
        }
        else if (Input.GetKeyDown(KeyCode.A))
        {
            Move(RoadDirection.Left);
        }
        else if (Input.GetKeyDown(KeyCode.S))
        {
            Move(RoadDirection.Down);
        }
        else if (Input.GetKeyDown(KeyCode.D))
        {
            Move(RoadDirection.Right);
        }
        tiles.tiles[tiles.tileRange / 2][tiles.tileRange / 2].color = (isClicked) ? waitColor : goColor;
    }
    private void Move(RoadDirection rd)
    {
        if (isClicked) return;
        bool isMoved = road.WalkRoad(rd);
        if (isMoved)
        {
            string label = (road.nowHistoryIndex > 3) ? road.history[road.nowHistoryIndex - 3].ToLabelString() + road.history[road.nowHistoryIndex - 2].ToLabelString() + road.history[road.nowHistoryIndex - 1].ToLabelString() : rd.ToLabelString();
            csv.Record("Marker",label);
            road.MakeNextRoad();
            Draw();
            StartCoroutine(WaitTime());
        }
    }
    private IEnumerator WaitTime()
    {
        isClicked = true;
        yield return new WaitForSeconds(waitTime);
        isClicked = false;
    }
    public void Draw()
    {
        tiles.ResetTileColor();
        var nowPos = new Vector2(tiles.tileRange / 2, tiles.tileRange / 2);
        //tiles.tiles[tiles.tileRange / 2][tiles.tileRange / 2].color = Color.red;
        for (int i = 1; i < 10; i++)
        {
            if (-1 == road.nowHistoryIndex - i) break;
            var d = road.history[road.nowHistoryIndex - i];
            nowPos += d.ToDirectionVec2() * -1;
            if (nowPos.x >= tiles.tileRange || nowPos.x < 0 || nowPos.y >= tiles.tileRange || nowPos.y < 0) continue;
            tiles.tiles[(int)nowPos.y][(int)nowPos.x].ChangeColor(afterColor);
        }
        nowPos = new Vector2(tiles.tileRange / 2, tiles.tileRange / 2);
        for (int i = 0; i < 10; i++)
        {
            if (road.history.Count == road.nowHistoryIndex + i) break;
            var d = road.history[road.nowHistoryIndex + i];
            nowPos += d.ToDirectionVec2();
            if (nowPos.x >= tiles.tileRange || nowPos.x < 0 || nowPos.y >= tiles.tileRange || nowPos.y < 0) continue;
            tiles.tiles[(int)nowPos.y][(int)nowPos.x].ChangeColor(beforeColor);
        }
    }
}
