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
        for (int i = 0; i < 5; i++) history.Add(RoadDirection.Down);
    }

    /**
     * �i�ނ��Ƃ��ł�����true���Ԃ�
     */
    public bool MakeNextRoad(RoadDirection rd)
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
            return RoadDirection.Right;//����
        }
        if (rd != history[nowHistoryIndex]) return false;
        nowHistoryIndex++;
        if (history[nowHistoryIndex - 1] == history[nowHistoryIndex]) return true;//���i�Ȃ�
        var newRoad = new RoadDirection[Random.Range(4, 10)];
        System.Array.Fill(newRoad,MakeNextDirection());
        history.AddRange(newRoad);
        return true;
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