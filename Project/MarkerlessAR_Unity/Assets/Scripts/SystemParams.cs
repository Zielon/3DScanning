using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

namespace Assets.Scripts
{
    public class SystemParams : MonoBehaviour
    {
        public InputField datesetPathInput;
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

            meshUpdateInput.text = "5";
            volumeSizeInput.text = "128";
            truncationInput.text = "7.0";
            datesetPathInput.text = absolutePath;
        }

        // Update is called once per frame
        private void Update()
        {
            PlayerPrefs.SetInt("volume_size", int.Parse(volumeSizeInput.text));
            PlayerPrefs.SetFloat("truncation", float.Parse(truncationInput.text));
            PlayerPrefs.SetInt("mesh_update", int.Parse(meshUpdateInput.text));
            PlayerPrefs.SetString("use_sensor", useSensorInput.isOn.ToString());

            var path = datesetPathInput.text;
            if (path.LastOrDefault() != Path.DirectorySeparatorChar ||
                path.LastOrDefault() != Path.AltDirectorySeparatorChar)
                path += Path.AltDirectorySeparatorChar;

            PlayerPrefs.SetString("dataset_path", path);
        }

        public void switchScene(string scene)
        {
            SceneManager.LoadScene(scene);
        }
    }
}