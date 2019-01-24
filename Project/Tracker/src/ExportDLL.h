#ifndef EXPORT_DLL_H
#define EXPORT_DLL_H

#include "TrackerContext.h"
#include "data-stream/headers/DatasetVideoStreamReader.h"
#include <opencv2/imgproc/imgproc.hpp>
#include "reconstruction/headers/Fusion.h"
#include "marshaling/__Mesh.h"
#include "data-stream/headers/Xtion2StreamReader.h"


//DLL exports of the m_tracker

#define OPENCV_TRAITS_ENABLE_DEPRECATED
//#define SENSOR_TEST

struct __MeshInfo;

extern "C" __declspec(dllexport) void* createContext(const char* dataset_path);

extern "C" __declspec(dllexport) void* createSensorContext();

extern "C" __declspec(dllexport) int getImageWidth(void* context);

extern "C" __declspec(dllexport) int getImageHeight(void* context);

extern "C" __declspec(dllexport) void tracker(void* context, unsigned char* image, float* pose);

extern "C" __declspec(dllexport) void getMeshInfo(void* context, __MeshInfo* info);

extern "C" __declspec(dllexport) void getMeshBuffers(__MeshInfo* _mesh_info, float* pVB, int* pIB);

#endif //EXPORT_DLL_H
