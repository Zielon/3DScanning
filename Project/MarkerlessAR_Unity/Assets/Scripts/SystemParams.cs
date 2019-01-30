﻿using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

namespace Assets.Scripts
{
    internal static class Settings
    {
        public const string VOLUME_SIZE = "volume_size";
        public const string TRUNCATION = "truncation";
        public const string MESH_UPDATE = "mesh_update";
        public const string USE_SENSOR = "use_sensor";
        public const string MAX_DEPTH = "max_depth";
        public const string DATASET_PATH = "dataset_path";
    }

    public class SystemParams : MonoBehaviour
    {
        public InputField datesetPathInput;
        public InputField maxDepthInput;
        public InputField meshUpdateInput;
        public InputField truncationInput;
        public Toggle useSensorInput;
        public InputField volumeSizeInput;

        // Start is called before the first frame update
        private void Start()
        {
            DontDestroyOnLoad(gameObject);

            var segments = new List<string>(
                    Application.dataPath.Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar))
                {"..", "Datasets", "freiburg", " "};

            var absolutePath = segments.Aggregate(
                (path, segment) => path += Path.AltDirectorySeparatorChar + segment).Trim();

            maxDepthInput.text = "4";
            meshUpdateInput.text = "5";
            volumeSizeInput.text = "128";
            truncationInput.text = "7.0";
            datesetPathInput.text = absolutePath;
        }

        // Update is called once per frame
        private void Update()
        {
            PlayerPrefs.SetInt(Settings.VOLUME_SIZE, int.Parse(volumeSizeInput.text));
            PlayerPrefs.SetFloat(Settings.TRUNCATION, float.Parse(truncationInput.text));
            PlayerPrefs.SetInt(Settings.MESH_UPDATE, int.Parse(meshUpdateInput.text));
            PlayerPrefs.SetString(Settings.USE_SENSOR, useSensorInput.isOn.ToString());
            PlayerPrefs.SetFloat(Settings.MAX_DEPTH, float.Parse(maxDepthInput.text));

            var path = datesetPathInput.text;
            if (path.LastOrDefault() != Path.DirectorySeparatorChar ||
                path.LastOrDefault() != Path.AltDirectorySeparatorChar)
                path += Path.AltDirectorySeparatorChar;

            PlayerPrefs.SetString(Settings.DATASET_PATH, path);
        }

        public void switchScene(string scene)
        {
            SceneManager.LoadScene(scene);
        }
    }
}