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
        for (int i = 0; i < 5; i++) history.Add(RoadDirection.Right);
        nowHistoryIndex = 0;
        for (int i = 0; i < 5; i++) history.Add(RoadDirection.Bottom);
    }

    /**
     * i‚Þ‚±‚Æ‚ª‚Å‚«‚½‚çtrue‚ª•Ô‚é
     */
    public bool MakeNextRoad(RoadDirection rd)
    {
        RoadDirection MakeNextDirection()
        {
            var r = Random.Range(0, 2);
            var last = history.Last();
            switch (last)
            {
                case RoadDirection.Top:
                case RoadDirection.Bottom:
                    return r == 0 ? RoadDirection.Left : RoadDirection.Right;
                case RoadDirection.Left:
                case RoadDirection.Right:
                    return r == 0 ? RoadDirection.Top : RoadDirection.Bottom;
            }
            return RoadDirection.Right;//–³Œø
        }
        if (rd != history[nowHistoryIndex]) return false;
        nowHistoryIndex++;
        if (history[nowHistoryIndex - 1] == history[nowHistoryIndex]) return true;//’¼i‚È‚ç
        var newRoad = new RoadDirection[Random.Range(4, 10)];
        System.Array.Fill(newRoad,MakeNextDirection());
        history.AddRange(newRoad);
        return true;
    }

}
public enum RoadDirection
{
    Top = 0, Bottom = 1, Left = 2, Right = 3
}
public static partial class EnumExtend
{
    public static Vector2 GetDirectionVec2(this RoadDirection param)
    {
        switch (param)
        {
            case RoadDirection.Top: return Vector2.up;
            case RoadDirection.Bottom: return Vector2.down;
            case RoadDirection.Left: return Vector2.left;
            case RoadDirection.Right: return Vector2.right;
        }
        return Vector2.zero;
    }
}