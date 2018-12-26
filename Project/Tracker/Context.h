#pragma once

#include "Tracker.h"
#include "VideoStreamReaderBase.h"

const std::string DATASET_DIR = "";
const bool ENFORCE_REALTIME = true;

struct Context {
    Tracker *tracker;
    VideoStreamReaderBase *videoStreamReader;
};