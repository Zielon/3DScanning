#include "Tracker.h"

void Tracker::computerCameraPose(byte *image, float *pose, int width, int height) {

    cout << "computerCameraPose" << endl;




    //cv::Mat matrix;

    //matrix.size();

    for (int i = 0; i < height; i++) {
        image[i] = (byte) 255;
    }
}



void Tracker::alignToNewFrame(cv::Mat& rgb, cv::Mat& depth, float * outPose)
{
	//todo: icp
}
