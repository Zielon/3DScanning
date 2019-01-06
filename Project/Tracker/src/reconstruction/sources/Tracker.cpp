#include "../headers/Tracker.h"

Tracker::Tracker() {
    m_icp = new ICP();
}

Tracker::~Tracker() {
    delete m_icp;
}

void Tracker::alignNewFrame(
        const std::vector<Vector3f> &source,
        const std::vector<Vector3f> &target, float *outPose) {

    auto pose = m_icp->estimatePose(source, target).data();

    for(int i = 0; i < 16; i++)
        outPose[i] = pose[i];

}

void Tracker::backprojectFrame(cv::Mat& depth, std::vector<Vector3f>& outVerts,const size_t pixelSteps)
{
	Vector3f pixelcoords;
	for (size_t y = 0; y < m_image_height; y+= pixelSteps)
	{
		for (size_t x = 0; x < m_image_width; x+= pixelSteps)
		{
			float depthVal = depth.at<float>(x,y)* 5000.0f; 
			if (depthVal>0.0f)
			{

				pixelcoords << (x - m_cX) / m_fovX * depthVal, (y - m_cY) / m_fovY * depthVal, depthVal;

				outVerts.push_back(pixelcoords); 

				//colIdx = idx * 4 * sizeof(BYTE);
				//vertices[idx].color << colorMap[colIdx], colorMap[colIdx + 1], colorMap[colIdx + 2], colorMap[colIdx + 3];

			}
			else
			{
				outVerts.push_back(Vector3f::Zero());
			}
		}
	}
}
