#ifndef PROJECT_TRACKER_H
#define PROJECT_TRACKER_H

#include <iostream>
#include <cstddef>

#include "ICP.h"
#include "../../data-stream/headers/VideoStreamReaderBase.h"
#include "../../data-stream/headers/DatasetVideoStreamReader.h"

#ifdef linux

#include <opencv2/core/mat.hpp>

#endif

#ifdef _WIN32

#include <opencv2/core.hpp>

#endif

using namespace std;

/**
 * Tracks frame to frame transition and estimate the post
 * It is assumed that a frame will be provided by a stream reader
 */
class Tracker {
public:
    Tracker();

    ~Tracker();

    void alignToNewFrame(cv::Mat &rgb, cv::Mat &depth, float *outPose);

private:
    ICP *m_icp;
};

#endif //PROJECT_TRACKER_H
