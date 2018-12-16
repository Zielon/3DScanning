#include "ExportDLL.h"

__declspec(dllexport) void * createTracker() {

	return new Tracker();
}

__declspec(dllexport) void trackerCameraPose(void *object, byte *image, float *pose, int w, int h) {

	Tracker *tracker = (Tracker*)object;

	tracker->computerCameraPose(image, pose, w, h);
}