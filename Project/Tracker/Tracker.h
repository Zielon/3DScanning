#ifndef PROJECT_TRACKER_H
#define PROJECT_TRACKER_H

#include <iostream>
#include <cstddef>

#ifdef linux

#include <opencv2/core/mat.hpp>

#endif

#ifdef _WIN32
// Add your import
#endif

using namespace std;

class Tracker {
public:
    void computerCameraPose(byte *image, float *pose, int width, int height);

private:
};

#endif //PROJECT_TRACKER_H
