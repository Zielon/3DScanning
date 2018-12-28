#ifndef EXPORT_DLL_H

#define EXPORT_DLL_H

#include "Context.h"
//DLL exports of the tracker

#ifdef _WIN32

extern "C" __declspec(dllexport) void * createContext();

extern "C" __declspec(dllexport) void trackerCameraPose(void *context, byte *image, float *pose, int w, int h);

extern "C" __declspec(dllexport) void dllMain(void *context, byte *image, float *pose);

extern "C" __declspec(dllexport) int getImageWidth(void *context);

extern "C" __declspec(dllexport) int getImageHeight(void *context);

#elif(__APPLE__)

extern "C" void * createContext();

extern "C" void trackerCameraPose(void *context, byte *image, float *pose, int w, int h);

extern "C" void dllMain(void *context, byte *image, float *pose);

extern "C" int getImageWidth(void *context);

extern "C" int getImageHeight(void *context);

#endif

#endif //EXPORT_DLL_H