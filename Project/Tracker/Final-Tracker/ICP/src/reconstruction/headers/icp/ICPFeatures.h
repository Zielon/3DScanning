#ifndef PROJECT_ICP_FEATURES_H
#define PROJECT_ICP_FEATURES_H
#include "ICP.h"

class ICPFeatures final : public ICP
{
public:
	ICPFeatures(SystemParameters parameters) : ICP(parameters){}

	~ICPFeatures(){}

	Matrix4f estimatePose(std::shared_ptr<PointCloud> model, std::shared_ptr<PointCloud> data) override;
};

#endif
