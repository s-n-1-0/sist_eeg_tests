using System.Collections;
using System.Collections.Generic;
using System.IO.Compression;
using System.Linq;
using UnityEngine;

public class RoadGenerator : MonoBehaviour
{   
    public List<RoadDirection> history = new List<RoadDirection>();
    [HideInInspector]
    public int nowHistoryIndex = 0;
    void Awake()
    {
        var newRoad = new RoadDirection[Random.Range(2, 6)];
        var r = Random.Range(0, 2);
        System.Array.Fill(newRoad, r == 0 ? RoadDirection.Left : RoadDirection.Right);
        history.AddRange(newRoad);
        newRoad = new RoadDirection[Random.Range(2, 6)];
        r = Random.Range(0, 2);
        System.Array.Fill(newRoad, r == 0 ? RoadDirection.Up : RoadDirection.Down);
        history.AddRange(newRoad);
    }

    /**
     * êiÇﬁÇ±Ç∆Ç™Ç≈Ç´ÇΩÇÁtrueÇ™ï‘ÇÈ
     */
    public bool WalkRoad(RoadDirection rd)
    {
        if (rd != history[nowHistoryIndex]) return false;
        nowHistoryIndex++;
        return true;//íºêiÇ»ÇÁ
    }
    public void MakeNextRoad()
    {
        RoadDirection MakeNextDirection()
        {
            var r = Random.Range(0, 2);
            var last = history.Last();
            switch (last)
            {
                case RoadDirection.Up:
                case RoadDirection.Down:
                    return r == 0 ? RoadDirection.Left : RoadDirection.Right;
                case RoadDirection.Left:
                case RoadDirection.Right:
                    return r == 0 ? RoadDirection.Up : RoadDirection.Down;
            }
            return RoadDirection.Right;//ñ≥å¯
        }
        if (history[nowHistoryIndex - 1] == history[nowHistoryIndex]) return;//íºêiÇ»ÇÁ
        var newRoad = new RoadDirection[Random.Range(2, 6)];
        System.Array.Fill(newRoad,MakeNextDirection());
        history.AddRange(newRoad);
    }

}
public enum RoadDirection
{
    Up = 0, Down = 1, Left = 2, Right = 3
}
public static partial class EnumExtend
{
    public static Vector2 ToDirectionVec2(this RoadDirection param)
    {
        switch (param)
        {
            case RoadDirection.Up: return Vector2.up;
            case RoadDirection.Down: return Vector2.down;
            case RoadDirection.Left: return Vector2.left;
            case RoadDirection.Right: return Vector2.right;
        }
        return Vector2.zero;
    }
    public static string ToLabelString(this RoadDirection param)
    {
        switch (param)
        {
            case RoadDirection.Up: return "T";
            case RoadDirection.Down: return "B";
            case RoadDirection.Left: return "L";
            case RoadDirection.Right: return "R";
        }
        return "";
    }
}