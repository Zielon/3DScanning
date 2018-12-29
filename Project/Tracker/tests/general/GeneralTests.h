#ifndef TRACKER_GENERALTESTS_H
#define TRACKER_GENERALTESTS_H

#include "../../reconstruction/source/NearestNeighbor.hpp"
#include "../../reconstruction/headers/ICP.h"

class GeneralTests {
public:
    void run();

private:
    void nearestNeighbor();
    void icp();
};


#endif //TRACKER_GENERALTESTS_H
