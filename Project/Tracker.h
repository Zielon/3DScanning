#ifndef PROJECT_TRACKER_H
#define PROJECT_TRACKER_H

#include <iostream>

using namespace std;

class Tracker{
public:
    void computerCameraPose(const float **image, const float *pose, int width, int height);
private:
};

#endif //PROJECT_TRACKER_H
