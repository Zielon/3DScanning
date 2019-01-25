#pragma once
#include <vtkAutoInit.h>
VTK_MODULE_INIT(vtkRenderingOpenGL);
//#include <boost/thread/thread.hpp>
#include <pcl/io/io.h>
#include <boost/make_shared.hpp>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/integral_image_normal.h>
//#include <pcl_visualization/cloud_viewer.h>

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
#include "../../reconstruction/headers/Mesh.h"
#include "../../concurency/headers/ThreadManager.h"

class TestBase
{
public:
	virtual ~TestBase() = default;

	TestBase();

	virtual void run() = 0;

protected:
	DatasetManager m_files_manager;

	Matrix4f getTrajectory(int index) const;

	static int getIterations();

	int wasKeyboardHit() const { return (int)_kbhit(); }

private:
	static std::vector<Matrix4f> m_trajectories;
	static std::vector<double> m_trajectory_timestamps;
	static std::vector<double> m_depth_timestamps;
};
