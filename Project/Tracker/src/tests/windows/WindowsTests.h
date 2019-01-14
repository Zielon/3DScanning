#ifndef TRACKER_WINDOWS_TESTS_H
#define TRACKER_WINDOWS_TESTS_H

#include "../../TrackerContext.h"
#include "../../reconstruction/headers/Tracker.h"
#include "../../data-stream/headers/DatasetVideoStreamReader.h"
#include "../../ExportDLL.h"
#include "../../files-manager/headers/DatasetManager.h"

class WindowsTests {
public:
    void run();

private:

	void streamPointCloudTest() const;

    void reconstructionTest() const;

    void vidReadTest();

	void cameraPoseTest();

	void meshTest(); 

	DatasetManager m_files_manager;

};

#endif //TRACKER_WINDOWSTESTS_H
