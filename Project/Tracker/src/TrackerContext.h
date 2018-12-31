#ifndef TRACKER_CONTEXT_H

#define TRACKER_CONTEXT_H

#include "reconstruction/headers/Tracker.h"
#include "data-stream/headers/VideoStreamReaderBase.h"

// Absolute path to the dataset dir, must end with a backslash
/*i.e.:
D:\\user\\desktop\\3dscanning\\assets\\rgbd_dataset_freiburg2_xyz\\
C:\\Users\\Lukas\\Desktop\\3DScanning\\Project\\Tracker\\data\\rgbd_dataset_freiburg1_xyz\\  
*/
const std::string DATASET_DIR = "";
const bool ENFORCE_REALTIME = true;

struct TrackerContext {
    Tracker *tracker;
    VideoStreamReaderBase *videoStreamReader;
};

#endif TRACKER_CONTEXT_H