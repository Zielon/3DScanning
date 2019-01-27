#ifndef PROJECT_ICP_CUDA_H
#define PROJECT_ICP_CUDA_H
#include "ICP.h"

class ICPCUDA final : public ICP
{
	~ICPCUDA() {}

	Matrix4f estimatePose(std::shared_ptr<PointCloud> model, std::shared_ptr<PointCloud> data) override;
};

#endif
