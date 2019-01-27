#include "../../headers/icp/ICPCUDA.h"

Matrix4f ICPCUDA::estimatePose(std::shared_ptr<PointCloud> model, std::shared_ptr<PointCloud> data){

	int threads = 224;
	int blocks = 96;

	m_icpOdom->initICPModel(model->m_depth_points);
	m_icpOdom->initICP(data->m_depth_points);

	T_wc_prev = T_wc_curr;

	Sophus::SE3d T_prev_curr = T_wc_prev.inverse() * T_wc_curr;

	m_icpOdom->getIncrementalTransformation(T_prev_curr, threads, blocks);

	T_wc_curr = T_wc_prev * T_prev_curr;

	return T_wc_curr.cast<float>().matrix();
}
