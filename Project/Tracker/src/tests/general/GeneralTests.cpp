#include "GeneralTests.h"

void GeneralTests::run() {
    nearestNeighbor();
    tracker();
    icp();
}

void GeneralTests::nearestNeighbor() {

    std::cout << "START nearestNeighbor()" << std::endl;

    auto nn = new NearestNeighborSearchFlann();

    nn->queryMatches({});
}

void GeneralTests::tracker() {

    std::cout << "START tracker()" << std::endl;

    std::vector<Vector3f> source;
    std::vector<Vector3f> target;

    source.emplace_back(Vector3f(0.3, 0.2, 0.1));
    source.emplace_back(Vector3f(0.4, 0.5, 0.3));
    source.emplace_back(Vector3f(0.5, 0.1, 0.7));

    target.emplace_back(Vector3f(0.1, 0.1, 0.1));
    target.emplace_back(Vector3f(0.3, 0.6, 0.4));
    target.emplace_back(Vector3f(0.3, 0.2, 0.1));

    auto tracker = new Tracker();

    float pose[16];

    tracker->alignNewFrame(source, target, pose);

    for(int i = 0; i < 16; i++) {
        if (i != 0 && i % 4 == 0)
            std::cout << std::endl;
        std::cout << pose[i] << " ";
    }

    std::cout << std::endl;
}

void GeneralTests::icp() {

    std::cout << "START icp() -> procrustes" << std::endl;

    auto procrustes = new ICP();

    std::vector<Vector3f> source;
    std::vector<Vector3f> target;

    source.emplace_back(Vector3f(0.3, 0.2, 0.1));
    source.emplace_back(Vector3f(0.4, 0.5, 0.3));
    source.emplace_back(Vector3f(0.5, 0.1, 0.7));

    target.emplace_back(Vector3f(0.1, 0.1, 0.1));
    target.emplace_back(Vector3f(0.3, 0.6, 0.4));
    target.emplace_back(Vector3f(0.3, 0.2, 0.1));

    auto pose = procrustes->estimatePose(source, target);

    std::cout << pose << std::endl;
}
