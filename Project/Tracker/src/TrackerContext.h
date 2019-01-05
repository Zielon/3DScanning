#ifndef TRACKER_CONTEXT_H

#define TRACKER_CONTEXT_H

#include "reconstruction/headers/Tracker.h"
#include "data-stream/headers/VideoStreamReaderBase.h"

// Absolute path to the dataset dir, must end with a backslash
/*i.e.:
D:\\user\\desktop\\3dscanning\\assets\\rgbd_dataset_freiburg2_xyz\\
C:\\Users\\\Lukas\\Documents\\3DScanning\\datasets\\rgbd_dataset_freiburg1_xyz\\
C:\\Users\\wojta\\Downloads\\rgbd_dataset_freiburg1_xyz\\rgbd_dataset_freiburg1_xyz\\
*/
const std::string DATASET_DIR = "C:\\Users\\\Lukas\\Documents\\3DScanning\\datasets\\rgbd_dataset_freiburg1_xyz\\";
const bool ENFORCE_REALTIME = true;

struct TrackerContext {
    Tracker *tracker;
    VideoStreamReaderBase *videoStreamReader;
};

#endif TRACKER_CONTEXT_H