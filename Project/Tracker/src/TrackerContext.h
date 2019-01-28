#ifndef TRACKER_CONTEXT_H
#define TRACKER_CONTEXT_H

#include "reconstruction/headers/Tracker.h"
#include "data-stream/headers/VideoStreamReaderBase.h"
#include "reconstruction/headers/FusionBase.h"
#include <list>


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
	FusionBase* m_fusion;
	bool m_first_frame = true; 
	bool enableReconstruction = true; 

	std::list<cv::Mat> rgb_recording; 
	std::list<cv::Mat> depth_recording; 

};

#endif TRACKER_CONTEXT_H
