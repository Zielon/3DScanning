#include <iostream>

#include "Tracker.h"

extern "C" __declspec(dllexport) int test() {
	return 8;
}

extern "C" __declspec(dllexport) void * createTracker() {
	
	return new Tracker();
}

extern "C" __declspec(dllexport) int trackerCount(void *object) {

	Tracker *tracker = (Tracker*) object;

	return tracker->count();
}

extern "C" __declspec(dllexport) void trackerCameraPose(void *object, byte *image, float *pose, int w, int h) {

	Tracker *tracker = (Tracker*)object;

	tracker->computerCameraPose(image, pose, w, h);

	//return tracker->count();
}

int main() {

    //Tracker tracker;

    //tracker.computerCameraPose(nullptr, nullptr, 0, 0);

    return 0;
}