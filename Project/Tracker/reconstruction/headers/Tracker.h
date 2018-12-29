#ifndef PROJECT_TRACKER_H
#define PROJECT_TRACKER_H

#include <iostream>
#include <cstddef>

#include "ICP.h"
#include "../../data-stream/headers/VideoStreamReaderBase.h"
#include "../../data-stream/headers/DatasetVideoStreamReader.h"

using namespace std;

/**
 * Tracks frame to frame transition and estimate the pose
 */
class Tracker {
public:
    Tracker();

    ~Tracker();

private:
    ICP *m_icp = nullptr;

    void alignToNewFrame(
            const std::vector<Vector3f> &sourcePoints,
            const std::vector<Vector3f> &targetPoints, float *outPose);
};

#endif //PROJECT_TRACKER_H
