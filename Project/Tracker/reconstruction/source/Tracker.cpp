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

    auto pose = m_icp->estimatePose(source, target).data();

    for(int i = 0; i < 16; i++)
        outPose[i] = pose[i];
}
