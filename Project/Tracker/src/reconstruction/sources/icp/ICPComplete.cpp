#include "../../headers/icp/ICPComplete.h"

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>

Matrix4f ICPComplete::estimatePose(std::shared_ptr<PointCloud> source, std::shared_ptr<PointCloud> target){

	std::vector<Vector3f> input = source->getPoints();
	std::vector<Vector3f> output = target->getPoints();

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZ>);

	cloud_in->points.resize(input.size());
	cloud_out->points.resize(output.size());

	for (int i = 0; i < input.size(); i++)
	{
		cloud_in->points[i].x = input[i].x();
		cloud_in->points[i].y = input[i].y();
		cloud_in->points[i].z = input[i].z();
	}

	for (int i = 0; i < output.size(); i++)
	{
		cloud_out->points[i].x = output[i].x();
		cloud_out->points[i].y = output[i].y();
		cloud_out->points[i].z = output[i].z();
	}

	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;

	icp.setMaxCorrespondenceDistance(0.025);
	icp.setMaximumIterations(50);
	icp.setTransformationEpsilon(1e-9);
	icp.setEuclideanFitnessEpsilon(0.8);
	icp.setInputSource(cloud_in);
	icp.setInputTarget(cloud_out);

	pcl::PointCloud<pcl::PointXYZ> Final;
	icp.align(Final);

	return icp.getFinalTransformation();
}
