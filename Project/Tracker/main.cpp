#include <iostream>

#include "Tracker.h"

//DLL exports of the tracker

__declspec(dllexport) void * createTracker() {
	
	return new Tracker();
}

__declspec(dllexport) void trackerCameraPose(void *object, byte *image, float *pose, int w, int h) {

	Tracker *tracker = (Tracker*)object;

	tracker->computerCameraPose(image, pose, w, h);
}

int main() {

    Tracker tracker;

    tracker.computerCameraPose(nullptr, nullptr, 0, 0);

    return 0;
}