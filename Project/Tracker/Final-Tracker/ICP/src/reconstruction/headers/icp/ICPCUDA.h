#ifndef PROJECT_ICP_CUDA_H
#define PROJECT_ICP_CUDA_H
#include "ICP.h"
#include "../../../icpcuda/ICPOdometry.h"

class ICPCUDA final : public ICP
{
public:
	ICPCUDA(){
		m_icpOdom = new ICPOdometry(640, 480, 319.5, 239.5, 528, 528);
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, 0);
		std::string dev(prop.name);
		std::cout << dev << std::endl;
	}

	~ICPCUDA(){
		SAFE_DELETE(m_icpOdom);
	}

	Matrix4f estimatePose(std::shared_ptr<PointCloud> model, std::shared_ptr<PointCloud> data) override;

	Matrix4f estimatePose(std::shared_ptr<PointCloud> model, std::shared_ptr<PointCloud> data, Matrix4f previous_pose) const;

private:
	ICPOdometry* m_icpOdom;
};

#endif
