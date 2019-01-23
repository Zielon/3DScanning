#pragma once
#ifndef PROJECT_ICP_H
#define PROJECT_ICP_H

#include "../../headers/PointCloud.h"

#include <opencv2/core.hpp>
#include <vector>

class ICP
{
public:

	virtual ~ICP(){};

	virtual Matrix4f estimatePose(std::shared_ptr<PointCloud> model, std::shared_ptr<PointCloud> data) = 0;
};

#endif //PROJECT_ICP_H
