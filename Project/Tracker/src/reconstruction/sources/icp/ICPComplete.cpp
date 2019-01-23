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
		cloud_in->points[i].getVector3fMap() = input[i];

	for (int i = 0; i < output.size(); i++)
		cloud_out->points[i].getVector3fMap() = output[i];

	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;

	icp.setMaxCorrespondenceDistance(50);//0.05
	//icp.setRANSACOutlierRejectionThreshold(0.05);
	icp.setRANSACIterations(1000);//50
	icp.setUseReciprocalCorrespondences(true);
	icp.setMaximumIterations(1000);//50
	icp.setTransformationEpsilon(1e-8);
	icp.setEuclideanFitnessEpsilon(1e-8);//0.1
	icp.setInputSource(cloud_in);
	icp.setInputTarget(cloud_out);

	pcl::PointCloud<pcl::PointXYZ> Final;
	icp.align(Final);

	return icp.getFinalTransformation();
}
