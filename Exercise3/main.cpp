#include <iostream>
#include <fstream>

#include "Eigen.h"
#include "SimpleMesh.h"
#include "ICPOptimizer.h"
#include "ProcrustesAligner.h"
#include "PointCloud.h"

#define USE_POINT_TO_PLANE	0

#define RUN_PROCRUSTES		0
#define RUN_SHAPE_ICP		1

void debugCorrespondenceMatching() {
	// Load the source and target mesh.
	const std::string filenameSource = PROJECT_DIR + std::string("/data/bunny/bunny_part1.off");
	const std::string filenameTarget = PROJECT_DIR + std::string("/data/bunny/bunny_part2_trans.off");

	SimpleMesh sourceMesh;
	if (!sourceMesh.loadMesh(filenameSource)) {
		std::cout << "Mesh file wasn't read successfully." << std::endl;
		return;
	}

	SimpleMesh targetMesh;
	if (!targetMesh.loadMesh(filenameTarget)) {
		std::cout << "Mesh file wasn't read successfully." << std::endl;
		return;
	}

	PointCloud source{ sourceMesh };
	PointCloud target{ targetMesh };
	
	// Search for matches using FLANN.
	std::unique_ptr<NearestNeighborSearch> nearestNeighborSearch = std::make_unique<NearestNeighborSearchFlann>();
	nearestNeighborSearch->setMatchingMaxDistance(0.0001f);
	nearestNeighborSearch->buildIndex(target.getPoints());
	auto matches = nearestNeighborSearch->queryMatches(source.getPoints());

	// Visualize the correspondences with lines.
	SimpleMesh resultingMesh = SimpleMesh::joinMeshes(sourceMesh, targetMesh, Matrix4f::Identity());
	auto sourcePoints = source.getPoints();
	auto targetPoints = target.getPoints();

	for (unsigned i = 0; i < 100; ++i) { // sourcePoints.size()
		const auto match = matches[i];
		if (match.idx >= 0) {
			const auto& sourcePoint = sourcePoints[i];
			const auto& targetPoint = targetPoints[match.idx];
			resultingMesh = SimpleMesh::joinMeshes(SimpleMesh::cylinder(sourcePoint, targetPoint, 0.002f, 2, 15), resultingMesh, Matrix4f::Identity());
		}
	}

	resultingMesh.writeMesh(PROJECT_DIR + std::string("/results/correspondences.off"));
}

int alignBunnyWithProcrustes() {
	// Load the source and target mesh.
	const std::string filenameSource = PROJECT_DIR + std::string("/data/bunny/bunny.off");
	const std::string filenameTarget = PROJECT_DIR + std::string("/data/bunny/bunny_trans.off");

	SimpleMesh sourceMesh;
	if (!sourceMesh.loadMesh(filenameSource)) {
		std::cout << "Mesh file wasn't read successfully at location: " << filenameSource << std::endl;
		return -1;
	}

	SimpleMesh targetMesh;
	if (!targetMesh.loadMesh(filenameTarget)) {
		std::cout << "Mesh file wasn't read successfully at location: " << filenameTarget << std::endl;
		return -1;
	}

	// Fill in the matched points: sourcePoints[i] is matched with targetPoints[i].
	std::vector<Vector3f> sourcePoints; 
	sourcePoints.push_back(Vector3f(-0.0106867f, 0.179756f, -0.0283248f)); // left ear
	sourcePoints.push_back(Vector3f(-0.0639191f, 0.179114f, -0.0588715f)); // right ear
	sourcePoints.push_back(Vector3f(0.0590575f, 0.066407f, 0.00686641f)); // tail
	sourcePoints.push_back(Vector3f(-0.0789843f, 0.13256f, 0.0519517f)); // mouth
	
	std::vector<Vector3f> targetPoints;
	targetPoints.push_back(Vector3f(-0.02744f, 0.179958f, 0.00980739f)); // left ear
	targetPoints.push_back(Vector3f(-0.0847672f, 0.180632f, -0.0148538f)); // right ear
	targetPoints.push_back(Vector3f(0.0544159f, 0.0715162f, 0.0231181f)); // tail
	targetPoints.push_back(Vector3f(-0.0854079f, 0.10966f, 0.0842135f)); // mouth
		
	// Estimate the pose from source to target mesh with Procrustes alignment.
	ProcrustesAligner aligner;
	Matrix4f estimatedPose = aligner.estimatePose(sourcePoints, targetPoints);

	// Visualize the resulting joined mesh. We add triangulated spheres for point matches.
	SimpleMesh resultingMesh = SimpleMesh::joinMeshes(sourceMesh, targetMesh, estimatedPose);
	for (const auto& sourcePoint : sourcePoints) {
		resultingMesh = SimpleMesh::joinMeshes(SimpleMesh::sphere(sourcePoint, 0.002f), resultingMesh, estimatedPose);
	}
	for (const auto& targetPoint : targetPoints) {
		resultingMesh = SimpleMesh::joinMeshes(SimpleMesh::sphere(targetPoint, 0.002f), resultingMesh, Matrix4f::Identity());
	}
	resultingMesh.writeMesh(PROJECT_DIR + std::string("/results/bunny_procrustes.off"));
	std::cout << "Resulting mesh written." << std::endl;
	
	return 0;
}

int alignBunnyWithICP() {
 	// Load the source and target mesh.
	const std::string filenameSource = PROJECT_DIR + std::string("/data/bunny/bunny_part1.off");
	const std::string filenameTarget = PROJECT_DIR + std::string("/data/bunny/bunny_part2_trans.off");

	SimpleMesh sourceMesh;
	if (!sourceMesh.loadMesh(filenameSource)) {
		std::cout << "Mesh file wasn't read successfully at location: " << filenameSource << std::endl;
		return -1;
	}

	SimpleMesh targetMesh;
	if (!targetMesh.loadMesh(filenameTarget)) {
		std::cout << "Mesh file wasn't read successfully at location: " << filenameTarget << std::endl;
		return -1;
	}

	// Estimate the pose from source to target mesh with ICP optimization.
	ICPOptimizer optimizer;
	optimizer.setMatchingMaxDistance(0.0003f);
	if (USE_POINT_TO_PLANE) {
		optimizer.usePointToPlaneConstraints(true);
		optimizer.setNbOfIterations(20);
	}
	else {
		optimizer.usePointToPlaneConstraints(false);
		optimizer.setNbOfIterations(25);
	}

	PointCloud source{ sourceMesh };
	PointCloud target{ targetMesh };

	Matrix4f estimatedPose = optimizer.estimatePose(source, target);
	
	// Visualize the resulting joined mesh. We add triangulated spheres for point matches.
	SimpleMesh resultingMesh = SimpleMesh::joinMeshes(sourceMesh, targetMesh, estimatedPose);
	resultingMesh.writeMesh(PROJECT_DIR + std::string("/results/bunny_icp.off"));
	std::cout << "Resulting mesh written." << std::endl;	

	return 0;
}

int main() {

	int result = 0;
	if (RUN_PROCRUSTES)
		result += alignBunnyWithProcrustes();
	if (RUN_SHAPE_ICP)
		result += alignBunnyWithICP();

	return result;
}
