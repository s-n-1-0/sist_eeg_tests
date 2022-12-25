using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

public class EndTestController : MonoBehaviour
{

    public Text countText;
    private void Start()
    {
        countText.text = "ãLò^âÒêî : " + TestController.testCount.ToString();
    }
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Q))
        {
            CsvExporter.standard.Close();
            #if UNITY_EDITOR
                        UnityEditor.EditorApplication.isPlaying = false;
            #else
                        Application.Quit();
            #endif
        }
        else if (Input.GetKeyDown(KeyCode.A))
        {
            SceneManager.LoadScene("TestScene");
        }
    }
}
