#include "Tracker.h"

void Tracker::computerCameraPose(byte *image, float *pose, int width, int height) {
    cout << "computerCameraPose" << endl;

	pose[0] = 5.0f;

	image[0] = (std::byte) 255;
}

int Tracker::count() {

	return -1;
}
