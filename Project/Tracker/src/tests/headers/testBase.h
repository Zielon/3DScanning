#pragma once

#include "../../TrackerContext.h"
#include "../../reconstruction/headers/Tracker.h"
#include "../../data-stream/headers/DatasetVideoStreamReader.h"
#include "../../ExportDLL.h"
#include "../../files-manager/headers/DatasetManager.h"

#include <fstream>
#include <sstream>
#include <direct.h>
#include <io.h>
#include "../../debugger/headers/Verbose.h"
#include "../../reconstruction/headers/Mesh.h"
#include "../../concurency/headers/ThreadManager.h"


class TestBase
{
public:
	TestBase();
	virtual void run() = 0;


protected:
	DatasetManager m_files_manager;
	static std::vector<Eigen::Matrix4f> m_trajectories;
};

