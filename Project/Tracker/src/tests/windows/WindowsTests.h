#ifndef TRACKER_WINDOWSTESTS_H
#define TRACKER_WINDOWSTESTS_H

#include "../../TrackerContext.h"
#include "../../reconstruction/headers/Tracker.h"
#include "../../data-stream/headers/DatasetVideoStreamReader.h"
#include "../../ExportDLL.h"

class WindowsTests {
public:
    void run();

private:
    void dllVidReadTest();

    void vidReadTest();
};

#endif //TRACKER_WINDOWSTESTS_H
