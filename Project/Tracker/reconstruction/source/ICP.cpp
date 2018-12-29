#include "../headers/ICP.h"

ICP::ICP() {
    m_nearestNeighbor = new NearestNeighborSearchFlann();
    m_procrustesAligner = new ProcrustesAligner();
}

ICP::~ICP() {
    delete m_nearestNeighbor;
    delete m_procrustesAligner;
}

Matrix4f ICP::estimatePose(const std::vector<Vector3f> &source, const std::vector<Vector3f> &target) {

    Matrix4f pose = Matrix4f::Identity();

    m_nearestNeighbor->buildIndex(target);

    std::vector<Vector3f> sourcePoints;
    std::vector<Vector3f> targetPoints;
    std::vector<Vector3f> targetNormals;

    for (int i = 0; i < m_number_iterations; ++i) {

        clock_t begin = clock();

        auto transformedPoints = transformPoints(source, pose);
        // TODO calculate normals for points
        //auto transformedNormals = transformNormals(source, pose);

        auto matches = m_nearestNeighbor->queryMatches(transformedPoints);

        // pruneCorrespondences(transformedNormals, target, matches);

        clock_t end = clock();
        double elapsedSecs = double(end - begin) / CLOCKS_PER_SEC;
        //std::cout << "Completed in " << elapsedSecs << " seconds." << std::endl;

        sourcePoints.clear();
        targetPoints.clear();
        targetNormals.clear();

        int numberOfMatches = 0;

        for (int j = 0; j < matches.size(); j++) {

            int idx = matches[j].idx; // Get source matches index

            if (idx <= -1) continue;

            numberOfMatches++;

            // Match exists
            sourcePoints.emplace_back(transformedPoints[j]);
            targetPoints.emplace_back(target[idx]);
            // targetNormals.emplace_back(transformedNormals[idx]);
        }

        if (numberOfMatches == 0) return pose;

        // TODO For now only point to point
        pose = estimatePosePointToPoint(sourcePoints, targetPoints) * pose;

    }

    return pose;
}

Matrix4f ICP::estimatePosePointToPoint(
        const std::vector<Vector3f> &sourcePoints,
        const std::vector<Vector3f> &targetPoints) {

    return m_procrustesAligner->estimatePose(sourcePoints, targetPoints);
}

void ICP::pruneCorrespondences(
        const std::vector<Vector3f> &sourceNormals,
        const std::vector<Vector3f> &targetNormals, std::vector<Match> &matches) {

    for (unsigned i = 0; i < sourceNormals.size(); i++) {

        Match match = matches[i];

        const auto &sourceNormal = sourceNormals[match.idx];
        const auto &targetNormal = targetNormals[match.idx];

        double normal_angle = std::acos(targetNormal.dot(sourceNormal)) * 180.0 / M_PI;

        if (normal_angle > 60.0f) {
            matches[i] = Match{-1, 0.f};
        }
    }
}

std::vector<Vector3f> ICP::transformPoints(const std::vector<Vector3f> &sourcePoints, const Matrix4f &pose) {
    std::vector<Vector3f> transformedPoints;
    transformedPoints.reserve(sourcePoints.size());

    const auto rotation = pose.block(0, 0, 3, 3);
    const auto translation = pose.block(0, 3, 3, 1);

    for (const auto &point : sourcePoints) {
        transformedPoints.emplace_back(rotation * point + translation);
    }

    return transformedPoints;
}

std::vector<Vector3f> ICP::transformNormals(const std::vector<Vector3f> &sourceNormals, const Matrix4f &pose) {
    std::vector<Vector3f> transformedNormals;
    transformedNormals.reserve(sourceNormals.size());

    const auto rotation = pose.block(0, 0, 3, 3);

    for (const auto &normal : sourceNormals) {
        transformedNormals.emplace_back(rotation.inverse().transpose() * normal);
    }

    return transformedNormals;
}