#ifndef PROJECT_ICP_H
#define PROJECT_ICP_H

#include <vector>

#include "../sources/NearestNeighbor.hpp"
#include "../sources/ProcrustesAligner.hpp"

#ifdef linux

#include <opencv2/core/mat.hpp>

#endif

#ifdef _WIN32

#define OPENCV_TRAITS_ENABLE_DEPRECATED
#include <opencv2/core.hpp>

#endif

class ICP {
public:
    ICP();

    ~ICP();

    Matrix4f estimatePose(const std::vector<Vector3f> &source, const std::vector<Vector3f> &target);

private:
    NearestNeighborSearch *m_nearestNeighbor;
    ProcrustesAligner *m_procrustesAligner;

    int m_number_iterations = 25;

    std::vector<Vector3f> transformPoints(const std::vector<Vector3f> &sourcePoints, const Matrix4f &pose);

    std::vector<Vector3f> transformNormals(const std::vector<Vector3f> &sourceNormals, const Matrix4f &pose);

    Matrix4f estimatePosePointToPoint(
            const std::vector<Vector3f> &sourcePoints,
            const std::vector<Vector3f> &targetPoints);

    void pruneCorrespondences(
            const std::vector<Vector3f> &sourceNormals,
            const std::vector<Vector3f> &targetNormals, std::vector<Match> &matches);

};

#endif //PROJECT_ICP_H
