#ifndef TRACKER_CONTEXT_H

#define TRACKER_CONTEXT_H

#include "reconstruction/headers/Tracker.h"
#include "data-stream/headers/VideoStreamReaderBase.h"
#include "reconstruction/headers/Fusion.h"

const bool ENFORCE_REALTIME = true;

struct TrackerContext {
    Tracker *m_tracker;
    VideoStreamReaderBase *m_videoStreamReader;
	Fusion* m_fusion; 
};

#endif TRACKER_CONTEXT_H
