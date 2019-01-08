#ifndef PROJECT_TRACKER_H
#define PROJECT_TRACKER_H

#include <iostream>
#include <cstddef>

#include "ICP.h"
#include "../../data-stream/headers/VideoStreamReaderBase.h"
#include "../../data-stream/headers/DatasetVideoStreamReader.h"

using namespace std;

/**
 * Tracks frame to frame transition and estimate the pose
 */
class Tracker {
public:
    Tracker();

	Tracker(float fovX, float fovY, float cx, float cy, int image_height, int image_width) :
		m_fovX(fovX), m_fovY(fovY), m_cX(cx), m_cY(cy),
		m_image_width(image_width), m_image_height(image_height)
	{
		m_icp = new ICP();
	}

    ~Tracker();

    void alignNewFrame(
            const std::vector<Vector3f> &sourcePoints,
            const std::vector<Vector3f> &targetPoints, float *outPose);

	void backprojectFrame(cv::Mat& depth, std::vector<Vector3f>& outVerts, const size_t pixelSteps = 1); 


	std::vector<Vector3f> m_previousFrameVerts; 
	ICP *m_icp = nullptr;

private:

	float m_fovX = 0, m_fovY = 0, m_cX = 0, m_cY = 0; 
	int m_image_height = 0, m_image_width = 0;
};

#endif //PROJECT_TRACKER_H
