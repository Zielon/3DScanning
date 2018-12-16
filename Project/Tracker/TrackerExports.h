#ifndef TRACKER_EXPORTS_H

#define TRACKER_EXPORTS_H

#include "Tracker.h"

//DLL exports of the tracker

extern "C" __declspec(dllexport) void * createTracker() {

	return new Tracker();
}

extern "C" __declspec(dllexport) void trackerCameraPose(void *object, byte *image, float *pose, int w, int h) {

	Tracker *tracker = (Tracker*)object;

	tracker->computerCameraPose(image, pose, w, h);
}


#endif //TRACKER_EXPORTS_H