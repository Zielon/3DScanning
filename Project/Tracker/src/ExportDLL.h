#ifndef EXPORT_DLL_H
#define EXPORT_DLL_H

#include "TrackerContext.h"
#include "data-stream/headers/DatasetVideoStreamReader.h"
#include <opencv2/imgproc/imgproc.hpp>
#include "reconstruction/headers/Fusion.h"
#include "reconstruction/headers/FusionGPU.h"
#include "marshaling/__Mesh.h"
#include "data-stream/headers/Xtion2StreamReader.h"
#include "marshaling/__SystemParameters.h"

//DLL exports of the m_tracker

#define OPENCV_TRAITS_ENABLE_DEPRECATED
//#define SENSOR_TEST

struct __MeshInfo;

extern "C" __declspec(dllexport) void* createContext(__SystemParameters* _parameters);

extern "C" __declspec(dllexport) void* createSensorContext(__SystemParameters* _parameters);

extern "C" __declspec(dllexport) int getImageWidth(void* context);

extern "C" __declspec(dllexport) int getImageHeight(void* context);

extern "C" __declspec(dllexport) void tracker(void* context, unsigned char* image, float* pose);

extern "C" __declspec(dllexport) void getFrame(void* context, unsigned char* image, bool record);

extern "C" __declspec(dllexport) void computeOfflineReconstruction(void* context, __MeshInfo* info, float* pose);

extern "C" __declspec(dllexport) void enableReconstruction(void* context, bool enable); 

extern "C" __declspec(dllexport) void getMeshInfo(void* context, __MeshInfo* info);

extern "C" __declspec(dllexport) void getMeshBuffers(__MeshInfo* _mesh_info, float* pVB, int* pIB);

extern "C" __declspec(dllexport) void deleteContext(void* context);


#endif //EXPORT_DLL_H
