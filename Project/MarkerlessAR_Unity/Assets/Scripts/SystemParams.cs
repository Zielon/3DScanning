using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using System.Threading;


namespace Assets.Scripts
{

    public class SystemParams : MonoBehaviour
    {
        public __SystemParameters m_systemParameters;

        public InputField truncationInput;
        public InputField volumeSizeInput;
        public InputField datesetPathInput;
        public Toggle useSensorInput; 

        public bool useSensor = false; 

        // Start is called before the first frame update
        void Start()
        {
            DontDestroyOnLoad(this.gameObject);

            var segments = new List<string>(
        Application.dataPath.Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar))
                {"..", "Datasets", "freiburg", " "};

            var absolutePath = segments.Aggregate(
                (path, segment) => path += Path.AltDirectorySeparatorChar + segment).Trim();

            volumeSizeInput.text = "256";
            truncationInput.text = "7.0";
            datesetPathInput.text = absolutePath; 
        }

        // Update is called once per frame
        void Update()
        {
            m_systemParameters.m_dataset_path = datesetPathInput.text;
            useSensor = useSensorInput.isOn; 
            int.TryParse(volumeSizeInput.text, out m_systemParameters.m_volume_size);
            float.TryParse(truncationInput.text, out m_systemParameters.m_truncation_scaling);

        }

        public void switchScene(string scene)
        {
            SceneManager.LoadScene(scene);
        }

    }
}