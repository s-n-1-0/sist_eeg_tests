using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;

public class CsvExporter
{
    public static string filename = $"./export_{DateTime.Now.ToString().Replace("/", "_").Replace(" ", "_").Replace(":", "")}";
    private StreamWriter sw;
    private System.Diagnostics.Stopwatch totalStopwatch = new System.Diagnostics.Stopwatch();
    private System.Diagnostics.Stopwatch syncStopwatch = new System.Diagnostics.Stopwatch();
    public CsvExporter()
    {
        sw = new StreamWriter($"{filename}_{TestController.testCount}.csv", true, Encoding.UTF8);
        string[] headers = { "Annotation","Label","Total Time", "Sync Elapsed Time" };
        sw.WriteLine(string.Join(",", headers));
        totalStopwatch.Start();
    }

    public void Record(string ann,string label)
    {
        if (ann == "sync") syncStopwatch.Restart();
        string[] s1 = { ann,label, totalStopwatch.Elapsed.TotalSeconds.ToString(), syncStopwatch.Elapsed.TotalSeconds.ToString() };
        string s2 = string.Join(",", s1);
        sw.WriteLine(s2);
    }
    public void Close()
    {
        sw.Close();
    }
}
