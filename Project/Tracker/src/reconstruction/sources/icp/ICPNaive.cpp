#include <iostream>
#include "../../../debugger/headers/Verbose.h"
#include "../../headers/icp/ICPNaive.h"

ICPNaive::ICPNaive(){
	m_procrustesAligner = new ProcrustesAligner();
}

ICPNaive::~ICPNaive(){
	SAFE_DELETE(m_nearestNeighbor);
	SAFE_DELETE(m_procrustesAligner);
}

Matrix4f ICPNaive::estimatePose(std::shared_ptr<PointCloud> source, std::shared_ptr<PointCloud> target){

	Matrix4f pose = Matrix4f::Identity();

	//std::cout << "Initial pose: " << std::endl;
	//std::cout << pose << std::endl;

	std::vector<Vector3f> sourcePoints;
	std::vector<Vector3f> targetPoints;
	std::vector<Vector3f> targetNormals;

	for (int i = 0; i < m_number_iterations; ++i)
	{
		auto transformedPoints = transformPoints(source->getPoints(), pose);
		auto transformedNormals = transformNormals(source->getNormals(), pose);

		auto matches = target->queryNearestNeighbor(transformedPoints);

		pruneCorrespondences(transformedNormals, target->getNormals(), matches);

		sourcePoints.clear();
		targetPoints.clear();
		targetNormals.clear();

		int numberOfMatches = 0;

		for (int j = 0; j < matches.size(); j++)
		{
			int idx = matches[j].idx; // Get source matches index

			if (idx <= -1) continue;

			numberOfMatches++;

			// Match exists
			sourcePoints.emplace_back(transformedPoints[j]);
			targetPoints.emplace_back(target->getPoints()[idx]);
			//targetNormals.emplace_back(transformedNormals[idx]);
			targetNormals.emplace_back(target->getNormals()[idx]);
		}

		if (numberOfMatches == 0)
		{
			Verbose::message("Aborted ICP: 0  valid matches", WARNING);

			return pose;
		}

		//pose = estimatePosePointToPoint(sourcePoints, targetPoints) * pose;
		pose = estimatePosePointToPlane(sourcePoints, targetPoints, targetNormals) * pose;
	}

	return pose;
}

Matrix4f ICPNaive::estimatePosePointToPoint(
	const std::vector<Vector3f>& sourcePoints,
	const std::vector<Vector3f>& targetPoints) const{

	return m_procrustesAligner->estimatePose(sourcePoints, targetPoints);
}

void ICPNaive::pruneCorrespondences(
	const std::vector<Vector3f>& sourceNormals,
	const std::vector<Vector3f>& targetNormals, std::vector<Match>& matches){

	for (unsigned i = 0; i < sourceNormals.size(); i++)
	{
		if(matches.empty()) return;

		Match match = matches[i];

		// Source and target vectors can have uneven number of elements
		// Matches are build from the target vector, therefore we have to check
		// if the source vector is still in the range
		if (match.idx >= sourceNormals.size())
		{
			matches[i] = Match{-1, 0.f};
			continue;
		}

		const auto& sourceNormal = sourceNormals[i];
		const auto& targetNormal = targetNormals[match.idx];

		double normal_angle = std::acos(targetNormal.dot(sourceNormal)) * 180.0 / M_PI;

		if (normal_angle > 60.0f)
		{
			matches[i] = Match{-1, 0.f};
		}
	}
}

std::vector<Vector3f> ICPNaive::transformPoints(const std::vector<Vector3f>& sourcePoints, const Matrix4f& pose) const{
	std::vector<Vector3f> transformedPoints;
	transformedPoints.reserve(sourcePoints.size());

	const auto rotation = pose.block(0, 0, 3, 3);
	const auto translation = pose.block(0, 3, 3, 1);

	for (const auto& point : sourcePoints)
	{
		transformedPoints.emplace_back(rotation * point + translation);
	}

	return transformedPoints;
}

std::vector<Vector3f> ICPNaive::transformNormals(const std::vector<Vector3f>& sourceNormals, const Matrix4f& pose){
	std::vector<Vector3f> transformedNormals;
	transformedNormals.reserve(sourceNormals.size());

	const auto rotation = pose.block(0, 0, 3, 3);

	for (const auto& normal : sourceNormals)
	{
		transformedNormals.emplace_back(rotation.inverse().transpose() * normal);
	}

	return transformedNormals;
}

Matrix4f ICPNaive::estimatePosePointToPlane(
	const std::vector<Vector3f>& sourcePoints,
	const std::vector<Vector3f>& targetPoints,
	const std::vector<Vector3f>& targetNormals) const{

	const unsigned nPoints = sourcePoints.size();

	// Build the system
	MatrixXf A = MatrixXf::Zero(4 * nPoints, 6);
	VectorXf b = VectorXf::Zero(4 * nPoints);

	for (unsigned i = 0; i < nPoints; i++)
	{
		const auto& s = sourcePoints[i];
		const auto& d = targetPoints[i];
		const auto& n = targetNormals[i];

		VectorXf nxs = s.cross(n);

		b(4 * i) = n.dot(d - s);
		A.row(4 * i).head<3>() = s.cross(n);
		A.row(4 * i).tail<3>() = n;

		b.segment(4 * i + 1, 3) = d - s;

		A.block(4 * i + 1, 0, 3, 6) <<
			0, s.z(), -s.y(), 1, 0, 0,
			-s.z(), 0, s.x(), 0, 1, 0,
			s.y(), -s.x(), 0, 0, 0, 1;
	}

	//std::cout << A << std::endl;
	//std::cout << b << std::endl;

	VectorXf x(6);

	JacobiSVD<MatrixXf> svd(A.transpose() * A, ComputeThinU | ComputeThinV);
	x = svd.solve(A.transpose() * b);

	float alpha = x(0), beta = x(1), gamma = x(2);

	// Build the pose matrix
	Matrix3f rotation = AngleAxisf(alpha, Vector3f::UnitX()).toRotationMatrix() *
		AngleAxisf(beta, Vector3f::UnitY()).toRotationMatrix() *
		AngleAxisf(gamma, Vector3f::UnitZ()).toRotationMatrix();

	Vector3f translation = x.tail(3);

	Matrix4f estimatedPose = Matrix4f::Identity();
	estimatedPose.block(0, 0, 3, 3) = rotation;
	estimatedPose.block(0, 3, 3, 1) = translation;

	//std::cout << estimatedPose << std::endl;

	return estimatedPose;
}
