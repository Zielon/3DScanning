#pragma once

#include "../../reconstruction/headers/Tracker.h"
#include "VideoStreamReaderBase.h"

// Absolute path to the dataset dir, must end with a backslash
/*i.e.:
D:\\user\\desktop\\3dscanning\\assets\\rgbd_dataset_freiburg2_xyz\\  
*/
const std::string DATASET_DIR = "D:\\user\\desktop\\3dscanning\\assets\\rgbd_dataset_freiburg2_xyz\\";
const bool ENFORCE_REALTIME = true;

struct Context {
    Tracker *tracker;
    VideoStreamReaderBase *videoStreamReader;
};