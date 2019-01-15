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
	}

	Tracker* m_tracker;
	VideoStreamReaderBase* m_videoStreamReader;
	Fusion* m_fusion;

	DatasetManager* m_datasetManager; 
	std::vector<Matrix4f> trajectories;


};


#endif TRACKER_CONTEXT_H
