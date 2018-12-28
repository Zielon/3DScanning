#ifndef PROJECT_ICP_H
#define PROJECT_ICP_H

#include "../source/NearestNeighbor.hpp"
#include "../source/PointCloud.hpp"

class ICP {
public:
    Matrix4f estimatePose(const PointCloud &source, const PointCloud &target, Matrix4f initialPose = Matrix4f::Identity());

private:

};

#endif //PROJECT_ICP_H
