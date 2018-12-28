#ifndef TRACKER_WINDOWSTESTS_H
#define TRACKER_WINDOWSTESTS_H

#include "../../dataStream/headers/Context.h"
#include "../../reconstruction/headers/Tracker.h"
#include "../../dataStream/headers/DatasetVideoStreamReader.h"
#include "../../ExportDLL.h"

class WindowsTests {
public:
    void run();

private:
    void dllVidReadTest();

    void vidReadTest();
};

#endif //TRACKER_WINDOWSTESTS_H
