#include "../headers/ICP.h"

ICP::ICP() {
    nearestNeighbor = new NearestNeighborSearchFlann();
}

ICP::~ICP() {
    delete nearestNeighbor;
}

Matrix4f ICP::estimatePose(const PointCloud &source, const PointCloud &target, Matrix4f initialPose) {

    //nearestNeighbor->buildIndex({});

    for (int i = 0; i < m_number_iterations; ++i) {
    }

    return Matrix4f();
}
