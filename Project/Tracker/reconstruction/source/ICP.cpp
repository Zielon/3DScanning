#include "../headers/ICP.h"

ICP::ICP() {
    m_nearestNeighbor = new NearestNeighborSearchFlann();
}

ICP::~ICP() {
    delete m_nearestNeighbor;
}

Matrix4f ICP::estimatePose(const std::vector<Vector3f> &source, const std::vector<Vector3f> &target, Matrix4f &pose) {

    m_nearestNeighbor->buildIndex(target);

    for (int i = 0; i < m_number_iterations; ++i) {

        // Compute the matches.
        std::cout << "Matching points ..." << std::endl;
        clock_t begin = clock();

        auto transformedPoints = transformPoints(source, pose);
        auto transformedNormals = transformNormals(source, pose);

        auto matches = m_nearestNeighbor->queryMatches(transformedPoints);
        pruneCorrespondences(transformedNormals, target, matches);

        clock_t end = clock();
        double elapsedSecs = double(end - begin) / CLOCKS_PER_SEC;
        std::cout << "Completed in " << elapsedSecs << " seconds." << std::endl;

        std::vector<Vector3f> sourcePoints;
        std::vector<Vector3f> targetPoints;
        std::vector<Vector3f> targetNormals;

        int numberOfMatches = 0;

        for (int j = 0; j < matches.size(); j++) {

            int idx = matches[j].idx; // Get source matches index

            if (idx <= -1) continue;

            numberOfMatches++;

            // Match exists
            sourcePoints.emplace_back(transformedPoints[j]);
            targetPoints.emplace_back(target[idx]);
            targetNormals.emplace_back(transformedNormals[idx]);
        }

        if (numberOfMatches == 0) return pose;

        pose = estimatePosePointToPoint(sourcePoints, targetPoints) * pose;

        std::cout << "Optimization iteration done." << std::endl;

    }

    return pose;
}

Matrix4f estimatePosePointToPoint(
        const std::vector<Vector3f> &sourcePoints,
        const std::vector<Vector3f> &targetPoints) {

    return Matrix4f();
}

void pruneCorrespondences(
        const std::vector<Vector3f> &sourceNormals,
        const std::vector<Vector3f> &targetNormals, std::vector<Match> &matches) {

    const unsigned nPoints = sourceNormals.size();

    for (unsigned i = 0; i < nPoints; i++) {

        Match match = matches[i];

        const auto &sourceNormal = sourceNormals[match.idx];
        const auto &targetNormal = targetNormals[match.idx];

        double normal_angle = std::acos(targetNormal.dot(sourceNormal)) * 180.0 / M_PI;

        if (normal_angle > 60.0f) {
            matches[i] = Match{-1, 0.f};
        }
    }
}

std::vector<Vector3f> transformPoints(const std::vector<Vector3f> &sourcePoints, const Matrix4f &pose) {
    std::vector<Vector3f> transformedPoints;
    transformedPoints.reserve(sourcePoints.size());

    const auto rotation = pose.block(0, 0, 3, 3);
    const auto translation = pose.block(0, 3, 3, 1);

    for (const auto &point : sourcePoints) {
        transformedPoints.emplace_back(rotation * point + translation);
    }

    return transformedPoints;
}

std::vector<Vector3f> transformNormals(const std::vector<Vector3f> &sourceNormals, const Matrix4f &pose) {
    std::vector<Vector3f> transformedNormals;
    transformedNormals.reserve(sourceNormals.size());

    const auto rotation = pose.block(0, 0, 3, 3);

    for (const auto &normal : sourceNormals) {
        transformedNormals.emplace_back(rotation.inverse().transpose() * normal);
    }

    return transformedNormals;
}