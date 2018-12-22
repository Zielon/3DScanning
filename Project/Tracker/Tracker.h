#ifndef PROJECT_TRACKER_H
#define PROJECT_TRACKER_H

#include <iostream>
#include <cstddef>

#ifdef linux

#include <opencv2/core/mat.hpp>

#endif

#ifdef _WIN32
#include <opencv2/core.hpp>
#endif




using namespace std;

class Tracker {
public:
    void computerCameraPose(byte *image, float *pose, int width, int height);

	void alignToNewFrame(cv::Mat& rgb, cv::Mat& depth, float * outPose);


private:
};

#endif //PROJECT_TRACKER_H
