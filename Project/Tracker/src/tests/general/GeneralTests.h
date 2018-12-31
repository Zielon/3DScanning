#ifndef TRACKER_GENERALTESTS_H
#define TRACKER_GENERALTESTS_H

#include "../../reconstruction/sources/NearestNeighbor.hpp"
#include "../../reconstruction/headers/ICP.h"
#include "../../reconstruction/headers/Tracker.h"

class GeneralTests {
public:
    void run();

private:
    void nearestNeighbor();
    void icp();
    void tracker();
};


#endif //TRACKER_GENERALTESTS_H
