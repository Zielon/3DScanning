#include "../headers/Tracker.h"

Tracker::Tracker() {
    m_icp = new ICP();
}

Tracker::~Tracker() {
    delete m_icp;
}

void Tracker::alignToNewFrame(cv::Mat &rgb, cv::Mat &depth, float *outPose) {


}
