#include "../../headers/icp/ICPCUDA.h"

Matrix4f ICPCUDA::estimatePose(std::shared_ptr<PointCloud> model, std::shared_ptr<PointCloud> data){

	int threads = 224;
	int blocks = 96;

	unsigned short* model_ = new unsigned short[640 * 480];
	unsigned short* data_ = new unsigned short[640 * 480];

	for(int i = 0; i < 640*480; i++)
	{
		model_[i] = ((unsigned short)model->m_depth_points[i]) / 5;
		data_[i] = ((unsigned short)data->m_depth_points[i]) / 5;
	}

	m_icpOdom->initICPModel(model_);
	m_icpOdom->initICP(data_);

	T_wc_prev = T_wc_curr;

	Sophus::SE3d T_prev_curr = T_wc_prev.inverse() * T_wc_curr;

	m_icpOdom->getIncrementalTransformation(T_prev_curr, threads, blocks);

	T_wc_curr = T_wc_prev * T_prev_curr;

	return T_wc_curr.cast<float>().matrix();
}
