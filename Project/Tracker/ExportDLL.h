#ifndef EXPORT_DLL_H

#define EXPORT_DLL_H

//Define data stream case
#define XTION_SENSOR
//#define DATASET

#include "TrackerContext.h"
#include <opencv2/imgproc/imgproc.hpp>

#ifdef DATASET

#include "DatasetVideoStreamReader.h"

#endif

#ifdef XTION_SENSOR

#include "XtionStreamReader.h"

#endif

/*Just include for test purposes
extern "C" __declspec(dllexport) int test();*/


//DLL exports of the tracker

#ifdef _WIN32

extern "C" __declspec(dllexport) void * createContext();

extern "C" __declspec(dllexport) void trackerCameraPose(void *context, byte *image, float *pose, int w, int h);

extern "C" __declspec(dllexport) int getImageWidth(void *context);

extern "C" __declspec(dllexport) int getImageHeight(void *context);

extern "C" __declspec(dllexport) void dllMain(void *context, byte *image, float *pose);

#endif

#endif //EXPORT_DLL_H