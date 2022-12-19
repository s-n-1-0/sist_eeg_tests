using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
public class TileImage : Image
{
    private Color wallColor = Color.black;

    protected override void Awake()
    {
        base.Awake();
        ChangeColor(null);
    }
    public void ChangeColor(Color? c)
    {
        this.color = c != null ? (Color)c : wallColor;
    }
}
