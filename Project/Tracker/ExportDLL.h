#ifndef EXPORT_DLL_H

#define EXPORT_DLL_H

#include "Tracker.h"

//DLL exports of the tracker

#ifdef _WIN32
extern "C" __declspec(dllexport) void * createTracker();

extern "C" __declspec(dllexport) void trackerCameraPose(void *object, byte *image, float *pose, int w, int h);
#endif

#endif //EXPORT_DLL_H