#ifndef TRACKER_WINDOWS_TESTS_H
#define TRACKER_WINDOWS_TESTS_H

#include "../../TrackerContext.h"
#include "../../reconstruction/headers/Tracker.h"
#include "../../data-stream/headers/DatasetVideoStreamReader.h"
#include "../../ExportDLL.h"
#include "../../files-manager/headers/DatasetManager.h"


const std::string CPP_DATASET_PATH ="\\..\\..\\..\\MarkerlessAR_Unity\\Datasets\\freiburg\\";

class WindowsTests {
public:
    void run();

private:

	void streamPointCloudTest();

    void reconstructionTest();

    void vidReadTest();

	bool cameraPoseTest();

	void meshTest(); 

	void precomputeMeshes();

	void WOzTest();

	DatasetManager m_files_manager;

};

#endif //TRACKER_WINDOWSTESTS_H
