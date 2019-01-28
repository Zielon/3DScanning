#ifndef PROJECT_ICP_CUDA_H
#define PROJECT_ICP_CUDA_H
#include "ICP.h"
#include "../../../icpcuda/ICPOdometry.h"

class ICPCUDA final : public ICP
{
public:
	ICPCUDA(SystemParameters parameters): ICP(parameters){

		m_icpOdom = new ICPOdometry(
			parameters.m_image_width,
			parameters.m_image_height,
			parameters.m_cX,
			parameters.m_cY,
			parameters.m_focal_length_X,
			parameters.m_focal_length_Y);

		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, 0);
		std::string dev(prop.name);
		std::cout << dev << std::endl;
		T_wc_prev = Sophus::SE3d();
		T_wc_curr = Sophus::SE3d();
	}

	~ICPCUDA(){
		SAFE_DELETE(m_icpOdom);
	}

	Matrix4f estimatePose(std::shared_ptr<PointCloud> model, std::shared_ptr<PointCloud> data) override;

private:
	ICPOdometry* m_icpOdom;
	Sophus::SE3d T_wc_prev;
	Sophus::SE3d T_wc_curr;
};

#endif
