#include "../headers/Tracker.h"

Tracker::Tracker() {
    m_icp = new ICP();
}

Tracker::~Tracker() {
    delete m_icp;
}

void Tracker::alignToNewFrame(
        const std::vector<Vector3f> &source,
        const std::vector<Vector3f> &target, float *outPose) {

    Matrix4f pose;
    m_icp->estimatePose(source, target, pose);
    outPose = pose.data();
}
