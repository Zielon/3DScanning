#include "../../headers/icp/ICPCUDA.h"

Matrix4f ICPCUDA::estimatePose(std::shared_ptr<PointCloud> model, std::shared_ptr<PointCloud> data){
	return Matrix4f::Identity();
}

Matrix4f ICPCUDA::estimatePose(std::shared_ptr<PointCloud> model, std::shared_ptr<PointCloud> data,
                               Matrix4f previous_pose) const{

	int threads = 512;
	int blocks = 512;

	m_icpOdom->initICPModel(reinterpret_cast<unsigned short *>(model->m_depth_points.data()));
	m_icpOdom->initICP(reinterpret_cast<unsigned short *>(data->m_depth_points.data()));

	const auto rotation = previous_pose.block(0, 0, 3, 3).cast<double>();
	const auto translation = previous_pose.block(0, 3, 3, 1).cast<double>();

	Sophus::SE3d T_wc_prev;

	T_wc_prev.rotationMatrix() = rotation;
	T_wc_prev.translation() = translation;

	Sophus::SE3d T_wc_curr = T_wc_prev;

	Sophus::SE3d T_prev_curr = T_wc_prev.inverse() * T_wc_curr;

	m_icpOdom->getIncrementalTransformation(T_prev_curr, threads, blocks);

	T_wc_curr = T_wc_prev * T_prev_curr;

	return Matrix<float, 4, 4>(T_wc_curr.cast<float>().matrix());
}
