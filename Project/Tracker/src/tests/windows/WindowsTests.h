#ifndef TRACKER_WINDOWSTESTS_H
#define TRACKER_WINDOWSTESTS_H

#include "../../TrackerContext.h"
#include "../../reconstruction/headers/Tracker.h"
#include "../../data-stream/headers/DatasetVideoStreamReader.h"
#include "../../ExportDLL.h"
#include "../../data-stream/headers/XtionStreamReader.h"
#include "../../data-stream/headers/Xtion2StreamReader.h"

#include <conio.h>

class WindowsTests {
public:
    void run();

private:

	bool readTrajectoryFile(const std::string& filename, std::vector<Eigen::Matrix4f>& result, std::vector<double>& timestamps);

	void readTrajectories();

    static void reconstructionTest();

    void vidReadTest();

	bool cameraPoseTest();

	void meshTest(); 

	int sensorTest(bool useOpenni2);
};

#endif //TRACKER_WINDOWSTESTS_H
