#ifndef PROJECT_ICP_COMPLETE_H
#define PROJECT_ICP_COMPLETE_H

#include <iostream>
#include "../PointCloud.h"
#include "ICP.h"

class ICPComplete final : public ICP
{
public:

	~ICPComplete(){}

	Matrix4f estimatePose(std::shared_ptr<PointCloud> source, std::shared_ptr<PointCloud> target) override;

private:
};

#endif //PROJECT_ICP_COMPLETE_H
