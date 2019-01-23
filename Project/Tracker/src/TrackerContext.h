#ifndef TRACKER_CONTEXT_H
#define TRACKER_CONTEXT_H

#include "reconstruction/headers/Tracker.h"
#include "data-stream/headers/VideoStreamReaderBase.h"
#include "reconstruction/headers/Fusion.h"

const bool ENFORCE_REALTIME = true;

struct TrackerContext
{
	~TrackerContext(){
		SAFE_DELETE(m_fusion);
		SAFE_DELETE(m_tracker);
		SAFE_DELETE(m_videoStreamReader);
	}

	Tracker* m_tracker;
	VideoStreamReaderBase* m_videoStreamReader;
	Fusion* m_fusion;
	bool m_first_frame = true; 
	int m_point_cloud_subsample_factor = 1; 
};

#endif TRACKER_CONTEXT_H
