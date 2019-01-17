#ifndef EXPORT_DLL_H

#define EXPORT_DLL_H
#include "TrackerContext.h"
#include "data-stream/headers/DatasetVideoStreamReader.h"
#include <opencv2/imgproc/imgproc.hpp>
#include "reconstruction/headers/Fusion.h"

//DLL exports of the m_tracker

#define OPENCV_TRAITS_ENABLE_DEPRECATED

struct __Mesh;

extern "C" __declspec(dllexport) void* createContext(const char* dataset_path);

extern "C" __declspec(dllexport) int getImageWidth(void* context);

extern "C" __declspec(dllexport) int getImageHeight(void* context);

extern "C" __declspec(dllexport) void tracker(void* context, unsigned char* image, float* pose);

extern "C" __declspec(dllexport) void getMesh(void* context, __Mesh* unity_mesh);

#endif //EXPORT_DLL_H
