using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
public class TileImage : Image
{
    private Color wallColor = Color.black;
    private Color roadColor = Color.white;

    protected override void Awake()
    {
        base.Awake();
        ChangeColor(false);
    }
    public void ChangeColor(bool isRoad)
    {
        this.color = isRoad ? roadColor : wallColor;
    }
}
