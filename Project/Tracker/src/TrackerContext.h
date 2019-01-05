#ifndef TRACKER_CONTEXT_H

#define TRACKER_CONTEXT_H

#include "reconstruction/headers/Tracker.h"
#include "data-stream/headers/VideoStreamReaderBase.h"

const bool ENFORCE_REALTIME = true;

struct TrackerContext {
    Tracker *tracker;
    VideoStreamReaderBase *videoStreamReader;
};

#endif TRACKER_CONTEXT_H