#include "../../headers/icp/ICPComplete.h"

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transformation_estimation_lm.h>
#include <pcl/registration/warp_point_rigid_3d.h>

typedef pcl::PointXYZ PointType;
typedef pcl::PointCloud<PointType> Cloud;
typedef Cloud::ConstPtr CloudConstPtr;
typedef Cloud::Ptr CloudPtr;

Matrix4f ICPComplete::estimatePose(std::shared_ptr<PointCloud> previous, std::shared_ptr<PointCloud> current){

	std::vector<Vector3f> current_points = current->getPoints();
	std::vector<Vector3f> previous_points = previous->getPoints();

	pcl::PointCloud<PointType>::Ptr model(new pcl::PointCloud<PointType>);
	pcl::PointCloud<PointType>::Ptr data(new pcl::PointCloud<PointType>);

	model->points.resize(previous_points.size());
	for (int i = 0; i < previous_points.size(); i++)
		model->points[i].getVector3fMap() = previous_points[i];

	data->points.resize(current_points.size());
	for (int i = 0; i < current_points.size(); i++)
		data->points[i].getVector3fMap() = current_points[i];

	pcl::IterativeClosestPointNonLinear<PointType, PointType> icp;

	boost::shared_ptr<pcl::registration::WarpPointRigid3D<PointType, PointType>> warp_fcn
		(new pcl::registration::WarpPointRigid3D<PointType, PointType>);

	boost::shared_ptr<pcl::registration::TransformationEstimationLM<PointType, PointType>> te(
		new pcl::registration::TransformationEstimationLM<PointType, PointType>);
	te->setWarpFunction(warp_fcn);

	icp.setTransformationEstimation(te);

	icp.setMaximumIterations(50);
	icp.setMaxCorrespondenceDistance(0.05);
	icp.setRANSACOutlierRejectionThreshold(0.05);

	icp.setInputTarget(model);
	icp.setInputSource(data);

	CloudPtr tmp(new Cloud);
	icp.align(*tmp);

	return icp.getFinalTransformation();
}
