#include "GeneralTests.h"

void GeneralTests::run() {
    nearestNeighbor();
}

void GeneralTests::nearestNeighbor() {

    std::cout << "START nearestNeighbor()" << std::endl;

    auto nn = new NearestNeighborSearchFlann();

    nn->buildIndex(nullptr);
}