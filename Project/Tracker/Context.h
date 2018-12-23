#pragma once


#include "Tracker.h"
#include "VideoStreamReader.h"


#ifdef IDK_HOW_TO_NOT_USE_ABS_PATH_IN_THIS_SITUATION
	const std::string DATASET_DIR = "D:/user/desktop/3dscanning/assets/rgbd_dataset_freiburg2_xyz/";
#else
	const std::string DATASET_DIR = "SETUP YOUR OWN DATASET PATH IN Context.h";
#endif
const bool ENFORCE_REALTIME = true;



struct Context
{
	Tracker* tracker; 
	VideoStreamReaderBase* videoStreamReader; 
};