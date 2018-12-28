#pragma once

#include "Tracker.h"
#include "VideoStreamReaderBase.h"

// Absolute path to the dataset dir, must end with a backslash
/*i.e.:
D:\\user\\desktop\\3dscanning\\assets\\rgbd_dataset_freiburg2_xyz\\
C:\\Users\\Lukas\\Desktop\\3DScanning\\Project\\Tracker\\data\\rgbd_dataset_freiburg1_xyz\\  
*/
const std::string DATASET_DIR = "C:\\Users\\Lukas\\Desktop\\3DScanning\\Project\\Tracker\\data\\rgbd_dataset_freiburg1_xyz\\";
const bool ENFORCE_REALTIME = true;

struct Context {
    Tracker *tracker;
    VideoStreamReaderBase *videoStreamReader;
};