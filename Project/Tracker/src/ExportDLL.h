#ifndef EXPORT_DLL_H

#define EXPORT_DLL_H
#include "Cpp14Workaround.h"
#include "TrackerContext.h"
#include "data-stream/headers/DatasetVideoStreamReader.h"
#include <opencv2/imgproc/imgproc.hpp>

//DLL exports of the tracker

#ifdef _WIN32

#define OPENCV_TRAITS_ENABLE_DEPRECATED

extern "C" __declspec(dllexport) void * createContext(char *dataset_path);

extern "C" __declspec(dllexport) void trackerCameraPose(void *context, unsigned char *image, float *pose, int w, int h);

extern "C" __declspec(dllexport) int getImageWidth(void *context);

extern "C" __declspec(dllexport) int getImageHeight(void *context);

extern "C" __declspec(dllexport) void dllMain(void *context, unsigned char *image, float *pose);

#endif

#endif //EXPORT_DLL_H