#ifndef TRACKER_CONTEXT_H
#define TRACKER_CONTEXT_H

#include "reconstruction/headers/Tracker.h"
#include "data-stream/headers/VideoStreamReaderBase.h"
#include "reconstruction/headers/Fusion.h"

const bool ENFORCE_REALTIME = true;

struct TrackerContext
{
	~TrackerContext(){
		SAFE_DELETE(m_tracker);
		SAFE_DELETE(m_videoStreamReader);
		SAFE_DELETE(m_fusion);
	}

	Tracker* m_tracker;
	VideoStreamReaderBase* m_videoStreamReader;
	Fusion* m_fusion;
};

struct WOzTrackerContext
{
	~WOzTrackerContext() {
		SAFE_DELETE(m_tracker);
		SAFE_DELETE(m_videoStreamReader);
		SAFE_DELETE(m_fusion);
		SAFE_DELETE(m_datasetManager); 
		SAFE_DELETE(currentMesh);
	}
	Tracker* m_tracker;
	VideoStreamReaderBase* m_videoStreamReader;
	Fusion* m_fusion;

	DatasetManager* m_datasetManager; 
	Mesh* currentMesh = nullptr; 
	Matrix4f invInitPose; 
	std::vector<Matrix4f> trajectories;
	std::vector<double> trajectory_timestamps;
	std::vector<double> depth_timestamps; 
	std::string meshPath; 
};


#endif TRACKER_CONTEXT_H
