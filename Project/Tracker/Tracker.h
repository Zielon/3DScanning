#ifndef PROJECT_TRACKER_H
#define PROJECT_TRACKER_H

#include <iostream>

//typedef unsigned char byte; 

/*Fix cmake file
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"*/

using namespace std;

class Tracker{
public:
    void computerCameraPose(byte *image, float *pose, int width, int height);

private:
};

//DLL exports
extern "C" __declspec(dllexport) void * createTracker();
extern "C" __declspec(dllexport) void trackerCameraPose(void *object, byte *image, float *pose, int w, int h);

#endif //PROJECT_TRACKER_H
