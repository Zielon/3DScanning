#include "Tracker.h"

void Tracker::computerCameraPose(byte *image, float *pose, int width, int height) {
    
	cout << "computerCameraPose" << endl;

	//Set image to white

	int N = height * width * 3;

	for (int i = 0; i < height; i++) {
		image[i] = (byte) 255;
	}
}