#pragma once


#include "Tracker.h"
#include "VideoStreamReader.h"

const std::string DATASET_DIR = "D:/user/desktop/3dscanning/assets/rgbd_dataset_freiburg2_xyz/";
const bool ENFORCE_REALTIME = true;



struct Context
{
	Tracker* tracker; 
	VideoStreamReaderBase* videoStreamReader; 
};