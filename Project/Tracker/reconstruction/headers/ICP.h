#ifndef PROJECT_ICP_H
#define PROJECT_ICP_H

#include "../source/NearestNeighbor.hpp"
#include "../source/PointCloud.hpp"

#ifdef linux

#include <opencv2/core/mat.hpp>

#endif

#ifdef _WIN32

#include <opencv2/core.hpp>

#endif

class ICP {
public:
    ICP();

    ~ICP();

    Matrix4f estimatePose(const PointCloud &source, const PointCloud &target, Matrix4f initialPose = Matrix4f::Identity());

private:
    NearestNeighborSearch *nearestNeighbor;
    int m_number_iterations = 25;
};

#endif //PROJECT_ICP_H
