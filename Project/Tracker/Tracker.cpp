#include "Tracker.h"

void Tracker::computerCameraPose(byte *image, float *pose, int width, int height) {
    cout << "computerCameraPose" << endl;

	//Simple test
	/*pose[0] = 5.0f;

	image[0] = (std::byte) 255;*/

	//Set image to white

	int N = height * width * 3;

	for (int i = 0; i < height; i++) {
		image[i] = (byte) 255;
	}
}

int Tracker::count() {

	return -1;
}
