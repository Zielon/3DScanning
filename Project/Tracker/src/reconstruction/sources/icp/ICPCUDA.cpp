#include "../../headers/icp/ICPCUDA.h"

Matrix4f ICPCUDA::estimatePose(std::shared_ptr<PointCloud> model, std::shared_ptr<PointCloud> data)
{
	//int threads = 224;
	//int blocks = 96;

	//Sophus::SE3d T_wc_prev;
	//Sophus::SE3d T_wc_curr;

	//ICPOdometry icpOdom(640, 480, 319.5, 239.5, 528, 528);

	//cudaDeviceProp prop;

	//cudaGetDeviceProperties(&prop, 0);

	//icpOdom.initICPModel((unsigned short *)model->m_depth_points.data());
	//icpOdom.initICP((unsigned short *)model->m_depth_points.data());

	//T_wc_prev = T_wc_curr;

	//Sophus::SE3d T_prev_curr = T_wc_prev.inverse() * T_wc_curr;

	//icpOdom.getIncrementalTransformation(T_prev_curr, threads, blocks);

	//T_wc_curr = T_wc_prev * T_prev_curr;

	return Matrix4f();
}
