#ifndef EXPORT_DLL_H

#define EXPORT_DLL_H
#include "TrackerContext.h"
#include "data-stream/headers/DatasetVideoStreamReader.h"
#include <opencv2/imgproc/imgproc.hpp>
#include "reconstruction/headers/Fusion.h"

//DLL exports of the m_tracker

#define OPENCV_TRAITS_ENABLE_DEPRECATED

extern "C" __declspec(dllexport) void * createContext(char *dataset_path);

extern "C" __declspec(dllexport) void trackerCameraPose(void *context, unsigned char *image, float *pose, int w, int h);

extern "C" __declspec(dllexport) int getImageWidth(void *context);

extern "C" __declspec(dllexport) int getImageHeight(void *context);

extern "C" __declspec(dllexport) void dllMain(void *context, unsigned char *image, float *pose);

extern "C" __declspec(dllexport) int getVertexCount(void* context);

extern "C" __declspec(dllexport) void getVertexBuffer(void* context, float *vertices); 

extern "C" __declspec(dllexport) void getNormalBuffer(void* context, float *normals);


extern "C" __declspec(dllexport) int getIndexCount(void* context);

extern "C" __declspec(dllexport) void getIndexBuffer(void* context, int* indices);


#endif //EXPORT_DLL_H
