#ifndef EXPORT_DLL_H

#define EXPORT_DLL_H

#include "Context.h"
#include "DatasetVideoStreamReader.h"
#include <opencv2/imgproc/imgproc.hpp>

//DLL exports of the tracker

//extern "C" __declspec(dllexport) int test();

#ifdef _WIN32

extern "C" __declspec(dllexport) void * createContext();

extern "C" __declspec(dllexport) void trackerCameraPose(void *context, byte *image, float *pose, int w, int h);

extern "C" __declspec(dllexport) int getImageWidth(void *context);

extern "C" __declspec(dllexport) int getImageHeight(void *context);

extern "C" __declspec(dllexport) void dllMain(void *context, byte *image, float *pose);

#endif

#endif //EXPORT_DLL_H