#ifndef TRACKER_LIB_FUSION_H
#define TRACKER_LIB_FUSION_H

#include "../../Eigen.h"

/**
 * Volumetric fusion class
 */
class Fusion {
public:
    void integrate(const std::vector<Vector3f> &cloud, Matrix4f &pose);
};

#endif //TRACKER_LIB_FUSION_H
