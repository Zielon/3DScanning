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
#include <ctime>
#include <chrono>
#include <iomanip>
#include <conio.h>
#include "../../debugger/headers/Verbose.h"
#include "../../debugger/headers/ProgressBar.hpp"
#include "../../reconstruction/headers/Mesh.h"
#include "../../concurency/headers/ThreadManager.h"

class TestBase
{
public:
	virtual ~TestBase() = default;

	TestBase();

	virtual void run() = 0;

	__SystemParameters* m_params;

protected:
	DatasetManager m_files_manager;

	Matrix4f getTrajectory(int index) const;

	static int getIterations();

	int wasKeyboardHit() const{
		return (int)_kbhit();
	}

private:
	static std::vector<Matrix4f> m_trajectories;
	static std::vector<double> m_trajectory_timestamps;
	static std::vector<double> m_depth_timestamps;
};
