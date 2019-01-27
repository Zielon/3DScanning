#include "../../headers/icp/ICPCUDA.h"

Matrix4f ICPCUDA::estimatePose(std::shared_ptr<PointCloud> model, std::shared_ptr<PointCloud> data){

	int threads = 512;
	int blocks = 96;

	Sophus::SE3d delta;

	cudaDeviceProp prop;

	cudaGetDeviceProperties(&prop, 0);

	m_icpOdom->initICPModel(reinterpret_cast<unsigned short *>(model->m_depth_points.data()));
	m_icpOdom->initICP(reinterpret_cast<unsigned short *>(data->m_depth_points.data()));
	m_icpOdom->getIncrementalTransformation(delta, threads, blocks);

	return Matrix<float, 4, 4>(delta.cast<float>().matrix());
}
