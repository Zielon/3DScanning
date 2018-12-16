#ifndef PROJECT_TRACKER_H
#define PROJECT_TRACKER_H

#include <iostream>

/*#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"*/

using namespace std;

class Tracker{
public:
    void computerCameraPose(const float **image, const float *pose, int width, int height);
	int count();

private:
};

#endif //PROJECT_TRACKER_H
