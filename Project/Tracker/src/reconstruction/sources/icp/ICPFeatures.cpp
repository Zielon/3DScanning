#include "../../headers/icp/ICPFeatures.h"

Matrix4f ICPFeatures::estimatePose(std::shared_ptr<PointCloud> model, std::shared_ptr<PointCloud> data){
	return Matrix4f::Identity();
}
