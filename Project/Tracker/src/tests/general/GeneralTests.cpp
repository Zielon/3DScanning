#include "GeneralTests.h"

void GeneralTests::run() {
    nearestNeighbor();
    tracker();
    icp();
}

void GeneralTests::nearestNeighbor() {

    std::cout << "START nearestNeighbor()" << std::endl;

    auto nn = new NearestNeighborSearchFlann();

	std::vector<Vector3f> vec;

	vec.emplace_back(Vector3f(0.3, 0.2, 0.1));

    nn->queryMatches(vec);
}

void GeneralTests::tracker() {

    std::cout << "START tracker()" << std::endl;

    auto tracker = new Tracker(CameraParameters());

    float pose[16];

    tracker->alignNewFrame(PointCloud(), PointCloud(), pose);

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



    auto pose = procrustes->estimatePose(PointCloud(), PointCloud());

    std::cout << pose << std::endl;
}
