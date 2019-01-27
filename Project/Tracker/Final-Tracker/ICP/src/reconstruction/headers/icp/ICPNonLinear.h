#ifndef PROJECT_ICP_COMPLETE_H
#define PROJECT_ICP_COMPLETE_H

#include <iostream>
#include "../PointCloud.h"
#include "ICP.h"

class ICPNonLinear final : public ICP
{
public:

	ICPNonLinear(SystemParameters parameters) : ICP(parameters) {}

	~ICPNonLinear(){}

	Matrix4f estimatePose(std::shared_ptr<PointCloud> model, std::shared_ptr<PointCloud> data) override;
};

#endif //PROJECT_ICP_COMPLETE_H
