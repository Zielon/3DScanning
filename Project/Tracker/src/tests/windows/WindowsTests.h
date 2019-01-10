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

	bool readTrajectoryFile(const std::string& filename, std::vector<Eigen::Matrix4f>& result, std::vector<double>& timestamps);

	void readTrajectories();

    void dllVidReadTest();

    void vidReadTest();

	bool cameraPoseTest();

	void meshTest(); 

};

#endif //TRACKER_WINDOWSTESTS_H
