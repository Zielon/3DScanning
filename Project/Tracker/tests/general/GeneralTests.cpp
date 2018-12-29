#include "GeneralTests.h"

void GeneralTests::run() {
    nearestNeighbor();
}

void GeneralTests::nearestNeighbor() {

    std::cout << "START m_nearestNeighbor()" << std::endl;

    auto nn = new NearestNeighborSearchFlann();

    nn->queryMatches({});
}