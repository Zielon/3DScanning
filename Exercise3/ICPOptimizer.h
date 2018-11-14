#pragma once

#include <flann/flann.hpp>

#include "SimpleMesh.h"
#include "NearestNeighbor.h"
#include "PointCloud.h"
#include "ProcrustesAligner.h"

#define M_PI           3.14159265358979323846  /* pi */

/**
 * ICP optimizer.
 */
class ICPOptimizer {
public:
	ICPOptimizer() : 
		m_bUsePointToPlaneConstraints{ false },
		m_nIterations{ 100 },
		m_nearestNeighborSearch{ std::make_unique<NearestNeighborSearchFlann>() }
	{ }

	void setMatchingMaxDistance(float maxDistance) {
		m_nearestNeighborSearch->setMatchingMaxDistance(maxDistance);
	}

	void usePointToPlaneConstraints(bool bUsePointToPlaneConstraints) {
		m_bUsePointToPlaneConstraints = bUsePointToPlaneConstraints;
	}

	void setNbOfIterations(unsigned nIterations) {
		m_nIterations = nIterations;
	}

	Matrix4f estimatePose(const PointCloud& source, const PointCloud& target, Matrix4f initialPose = Matrix4f::Identity()) {
		// Build the index of the FLANN tree (for fast nearest neighbor lookup).
		m_nearestNeighborSearch->buildIndex(target.getPoints());

		// The initial estimate can be given as an argument.
		Matrix4f estimatedPose = initialPose;

		for (int i = 0; i < m_nIterations; ++i) {
			// Compute the matches.
			std::cout << "Matching points ..." << std::endl;
			clock_t begin = clock();

			auto transformedPoints = transformPoints(source.getPoints(), estimatedPose);
			auto transformedNormals = transformNormals(source.getNormals(), estimatedPose);

			auto matches = m_nearestNeighborSearch->queryMatches(transformedPoints);
			pruneCorrespondences(transformedNormals, target.getNormals(), matches);

			clock_t end = clock();
			double elapsedSecs = double(end - begin) / CLOCKS_PER_SEC;
			std::cout << "Completed in " << elapsedSecs << " seconds." << std::endl;

			std::vector<Vector3f> sourcePoints;
			std::vector<Vector3f> targetPoints;
			std::vector<Vector3f> targetNormals;

			// TODO: Add all matches to the sourcePoints and targetPoints vector, so that
			//       sourcePoints[i] matches targetPoints[i]. For every source point, the
			//       'matches' vector holds the index of the matching target point and target normal.

			auto source_points = source.getPoints();
			auto target_points = target.getPoints();
			auto target_normals = target.getNormals();

			int numberOfMatches = 0;

			for(int j = 0; j < matches.size(); j++){

			    int idx = matches[j].idx; // Get source matches index

				if(idx <= -1) continue;

                numberOfMatches++;

				// Match exists
                sourcePoints.emplace_back(source_points[idx]);
                targetPoints.emplace_back(target_points[idx]);
                targetNormals.emplace_back(target_normals[idx]);
			}

			std::cout << std::endl << "Number of matches: " << numberOfMatches << " iteration: " << i + 1 << std::endl << std::endl;

			if(numberOfMatches == 0) return estimatedPose;

			// Estimate the new pose
 			if (m_bUsePointToPlaneConstraints) {
				estimatedPose = estimatePosePointToPlane(sourcePoints, targetPoints, targetNormals) * estimatedPose;
			}
			else {
				estimatedPose = estimatePosePointToPoint(sourcePoints, targetPoints) * estimatedPose;
			}

			std::cout << "Optimization iteration done." << std::endl;
		}

		return estimatedPose;
	}

private:
	bool m_bUsePointToPlaneConstraints;
	unsigned m_nIterations;
	std::unique_ptr<NearestNeighborSearch> m_nearestNeighborSearch;

	std::vector<Vector3f> transformPoints(const std::vector<Vector3f>& sourcePoints, const Matrix4f& pose) {
		std::vector<Vector3f> transformedPoints;
		transformedPoints.reserve(sourcePoints.size());

		const auto rotation = pose.block(0, 0, 3, 3);
		const auto translation = pose.block(0, 3, 3, 1);

		for (const auto& point : sourcePoints) {
			transformedPoints.push_back(rotation * point + translation);
		}

		return transformedPoints;
	}

	std::vector<Vector3f> transformNormals(const std::vector<Vector3f>& sourceNormals, const Matrix4f& pose) {
		std::vector<Vector3f> transformedNormals;
		transformedNormals.reserve(sourceNormals.size());

		const auto rotation = pose.block(0, 0, 3, 3);

		for (const auto& normal : sourceNormals) {
			transformedNormals.push_back(rotation.inverse().transpose() * normal);
		}

		return transformedNormals;
	}

	void pruneCorrespondences(const std::vector<Vector3f>& sourceNormals, const std::vector<Vector3f>& targetNormals, std::vector<Match>& matches) {
		const unsigned nPoints = sourceNormals.size();

		for (unsigned i = 0; i < nPoints; i++) {

		    Match match = matches[i];
            const auto& sourceNormal = sourceNormals[match.idx];
            const auto& targetNormal = targetNormals[match.idx];

            // TODO: Invalidate the match (set it to -1) if the angle between the normals is greater than 60

            double normal_angle = std::acos(targetNormal.dot(sourceNormal)) * 180.0 / M_PI;

            if (normal_angle > 60.0){
                matches[i] = Match{ -1, 0.f };
            }
		}
	}

	Matrix4f estimatePosePointToPoint(const std::vector<Vector3f>& sourcePoints, const std::vector<Vector3f>& targetPoints) {
		ProcrustesAligner procrustAligner;
		Matrix4f estimatedPose = procrustAligner.estimatePose(sourcePoints, targetPoints);

		return estimatedPose;
	}

	Matrix4f estimatePosePointToPlane(const std::vector<Vector3f>& sourcePoints, const std::vector<Vector3f>& targetPoints, const std::vector<Vector3f>& targetNormals) {
		const unsigned nPoints = sourcePoints.size();

		// Build the system
		//MatrixXf A = MatrixXf::Zero(4 * nPoints, 6);
		//VectorXf b = VectorXf::Zero(4 * nPoints);

		MatrixXf A = MatrixXf::Zero(nPoints, 6);
		VectorXf b = VectorXf::Zero(nPoints);
		//std::cout << sourcePoints.size() <<std::endl;

		for (unsigned i = 0; i < nPoints; i++) {
			const auto& s = sourcePoints[i];
			const auto& d = targetPoints[i];
			const auto& n = targetNormals[i];

			// TODO: Add the point-to-plane constraints to the system

			VectorXf nxs = s.cross(n);

			for(int j = 0; j < 3; j++){
				A(i, j) = nxs[j];
                A(i, j+3) = n[j];
			}

			b(i) = n.dot(d) - n.dot(s);

			// TODO: Add the point-to-point constraints to the system

		}

		//std::cout << A << std::endl;
		//std::cout << b << std::endl;

		// TODO: Solve the system
		VectorXf x(6);

		JacobiSVD<MatrixXf> svd(A.transpose() * A, ComputeThinU | ComputeThinV);
		x = svd.solve(A.transpose() * b);


		//Manual approach
		//MatrixXf Ainv = svd.matrixV() * svd.matrixU().transpose();//Pseudo inverse using SVD decomposition
		//x = Ainv.transpose()*b;
		
		float alpha = x(0), beta = x(1), gamma = x(2);

		// Build the pose matrix
		Matrix3f rotation = AngleAxisf(alpha, Vector3f::UnitX()).toRotationMatrix() *
			                AngleAxisf(beta, Vector3f::UnitY()).toRotationMatrix() *
			                AngleAxisf(gamma, Vector3f::UnitZ()).toRotationMatrix();

		Vector3f translation = x.tail(3);

		// TODO: Build the pose matrix using the rotation and translation matrices
		Matrix4f estimatedPose = Matrix4f::Identity();
		estimatedPose.block(0, 0, 3, 3) = rotation;
		estimatedPose.block(0, 3, 3, 1) = translation;

		std::cout << estimatedPose << std::endl;

		return estimatedPose;
	}
};
