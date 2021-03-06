#include "../../headers/icp/ICPNonLinear.h"

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transformation_estimation_lm.h>
#include <pcl/registration/warp_point_rigid_3d.h>

typedef pcl::PointXYZ PointType;
typedef pcl::PointXYZRGBNormal PointNormalT;
typedef pcl::PointCloud<PointType> Cloud;
typedef Cloud::ConstPtr CloudConstPtr;
typedef Cloud::Ptr CloudPtr;

void addNormal(pcl::PointCloud<PointType>::Ptr cloud,
               pcl::PointCloud<PointNormalT>::Ptr cloud_with_normals
){
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

	pcl::search::KdTree<pcl::PointXYZ>::Ptr searchTree(new pcl::search::KdTree<pcl::PointXYZ>);
	searchTree->setInputCloud(cloud);

	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimator;
	normalEstimator.setInputCloud(cloud);
	normalEstimator.setSearchMethod(searchTree);
	normalEstimator.setKSearch(5);
	normalEstimator.compute(*normals);

	concatenateFields(*cloud, *normals, *cloud_with_normals);
}

Matrix4f ICPNonLinear::estimatePose(std::shared_ptr<PointCloud> previous, std::shared_ptr<PointCloud> current){

	std::vector<Vector3f> current_points = current->getPoints();
	std::vector<Vector3f> previous_points = previous->getPoints();

	pcl::PointCloud<PointType>::Ptr model(new pcl::PointCloud<PointType>);
	pcl::PointCloud<PointType>::Ptr data(new pcl::PointCloud<PointType>);

	for (int i = 0; i < previous_points.size(); i += 16)
		model->points.push_back(pcl::PointXYZ(previous_points[i].x(), previous_points[i].y(), previous_points[i].z()));

	for (int i = 0; i < current_points.size(); i += 16)
		data->points.push_back(pcl::PointXYZ(current_points[i].x(), current_points[i].y(), current_points[i].z()));

	const pcl::PointCloud<PointNormalT>::Ptr model_normal(new pcl::PointCloud<PointNormalT>);
	const pcl::PointCloud<PointNormalT>::Ptr data_normal(new pcl::PointCloud<PointNormalT>);

	addNormal(model, model_normal);
	addNormal(data, data_normal);

	// Levenberg�Marquardt algorithm
	pcl::IterativeClosestPoint<PointNormalT, PointNormalT> icp;

	const boost::shared_ptr<pcl::registration::WarpPointRigid3D<PointNormalT, PointNormalT>> warp_fcn
		(new pcl::registration::WarpPointRigid3D<PointNormalT, PointNormalT>);

	boost::shared_ptr<pcl::registration::TransformationEstimationLM<PointNormalT, PointNormalT>> te(
		new pcl::registration::TransformationEstimationLM<PointNormalT, PointNormalT>);
	te->setWarpFunction(warp_fcn);

	icp.setTransformationEstimation(te);
	icp.setMaximumIterations(50);
	icp.setMaxCorrespondenceDistance(0.04);
	icp.setRANSACOutlierRejectionThreshold(0.04);
	icp.setTransformationEpsilon(1e-8);

	icp.setInputTarget(model_normal);
	icp.setInputSource(data_normal);

	pcl::PointCloud<PointNormalT>::Ptr tmp(new pcl::PointCloud<PointNormalT>);
	icp.align(*tmp);

	return icp.getFinalTransformation();
}
