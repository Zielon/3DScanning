#include "../../headers/icp/ICPFeatures.h"

#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/time.h>
#include <pcl/console/print.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/segmentation/sac_segmentation.h>

typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointNT> PointCloudT;
typedef pcl::FPFHSignature33 FeatureT;
typedef pcl::FPFHEstimationOMP<PointNT, PointNT, FeatureT> FeatureEstimationT;
typedef pcl::PointCloud<FeatureT> FeatureCloudT;

Matrix4f ICPFeatures::estimatePose(std::shared_ptr<PointCloud> previous, std::shared_ptr<PointCloud> current){

	std::vector<Vector3f> current_points = current->getPoints();
	std::vector<Vector3f> previous_points = previous->getPoints();

	PointCloudT::Ptr object(new PointCloudT);
	PointCloudT::Ptr object_aligned(new PointCloudT);
	PointCloudT::Ptr scene(new PointCloudT);
	FeatureCloudT::Ptr object_features(new FeatureCloudT);
	FeatureCloudT::Ptr scene_features(new FeatureCloudT);

	for (int i = 0; i < previous_points.size(); i += 1)
	{
		pcl::PointNormal point;
		point.x = previous_points[i].x();
		point.y = previous_points[i].y();
		point.z = previous_points[i].z();
		scene->points.push_back(point);
	}

	for (int i = 0; i < current_points.size(); i += 1)
	{
		pcl::PointNormal point;
		point.x = current_points[i].x();
		point.y = current_points[i].y();
		point.z = current_points[i].z();
		object->points.push_back(point);
	}

	pcl::NormalEstimationOMP<PointNT, PointNT> nest;
	nest.setRadiusSearch(0.01);
	nest.setInputCloud(scene);
	nest.compute(*scene);

	FeatureEstimationT fest;
	fest.setRadiusSearch(0.025);
	fest.setInputCloud(object);
	fest.setInputNormals(object);
	fest.compute(*object_features);
	fest.setInputCloud(scene);
	fest.setInputNormals(scene);
	fest.compute(*scene_features);

	pcl::SampleConsensusPrerejective<PointNT, PointNT, FeatureT> align;
	align.setInputSource(object);
	align.setSourceFeatures(object_features);
	align.setInputTarget(scene);
	align.setTargetFeatures(scene_features);
	align.setMaximumIterations(50000);		// Number of RANSAC iterations
	align.setNumberOfSamples(3);			// Number of points to sample for generating/prerejecting a pose
	align.setCorrespondenceRandomness(5); // Number of nearest features to use
	align.setSimilarityThreshold(0.9f);		// Polygonal edge length similarity threshold
	align.setInlierFraction(0.25f);			// Required inlier fraction for accepting a pose hypothesis
	align.align(*object_aligned);

	return align.getFinalTransformation();
}
