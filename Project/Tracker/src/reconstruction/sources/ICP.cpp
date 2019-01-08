#include "../headers/ICP.h"

ICP::ICP(){
	m_nearestNeighbor = new NearestNeighborSearchFlann();
	m_procrustesAligner = new ProcrustesAligner();
}

ICP::~ICP(){
	delete m_nearestNeighbor;
	delete m_procrustesAligner;
}

Matrix4f ICP::estimatePose(const PointCloud& source, const PointCloud& target){

	Matrix4f pose = Matrix4f::Identity();

	m_nearestNeighbor->buildIndex(target.getPoints());

	std::vector<Vector3f> sourcePoints;
	std::vector<Vector3f> targetPoints;
	std::vector<Vector3f> targetNormals;

	for (int i = 0; i < m_number_iterations; ++i)
	{
		clock_t begin = clock();

		auto transformedPoints = transformPoints(source.getPoints(), pose);
		auto transformedNormals = transformNormals(source.getNormals(), pose);

		auto matches = m_nearestNeighbor->queryMatches(transformedPoints);

		//pruneCorrespondences(transformedNormals, target.getNormals(), matches);

		clock_t end = clock();
		double elapsedSecs = double(end - begin) / CLOCKS_PER_SEC;
		std::cout << "Completed in " << elapsedSecs << " seconds." << std::endl;

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
			targetPoints.emplace_back(target.getPoints()[idx]);
			//targetNormals.emplace_back(transformedNormals[idx]);
		}

		if (numberOfMatches == 0) return pose;

		pose = estimatePosePointToPoint(sourcePoints, targetPoints) * pose;
		//pose = estimatePosePointToPlane(sourcePoints, targetPoints, targetNormals) * pose;
	}

	return pose;
}

Matrix4f ICP::estimatePosePointToPoint(
	const std::vector<Vector3f>& sourcePoints,
	const std::vector<Vector3f>& targetPoints){

	return m_procrustesAligner->estimatePose(sourcePoints, targetPoints);
}

void ICP::pruneCorrespondences(
	const std::vector<Vector3f>& sourceNormals,
	const std::vector<Vector3f>& targetNormals, std::vector<Match>& matches){

	for (unsigned i = 0; i < sourceNormals.size(); i++)
	{
		Match match = matches[i];

		const auto& sourceNormal = sourceNormals[match.idx];
		const auto& targetNormal = targetNormals[match.idx];

		double normal_angle = std::acos(targetNormal.dot(sourceNormal)) * 180.0 / M_PI;

		if (normal_angle > 60.0f)
		{
			matches[i] = Match{-1, 0.f};
		}
	}
}

std::vector<Vector3f> ICP::transformPoints(const std::vector<Vector3f>& sourcePoints, const Matrix4f& pose){
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

std::vector<Vector3f> ICP::transformNormals(const std::vector<Vector3f>& sourceNormals, const Matrix4f& pose){
	std::vector<Vector3f> transformedNormals;
	transformedNormals.reserve(sourceNormals.size());

	const auto rotation = pose.block(0, 0, 3, 3);

	for (const auto& normal : sourceNormals)
	{
		transformedNormals.emplace_back(rotation.inverse().transpose() * normal);
	}

	return transformedNormals;
}

Matrix4f ICP::estimatePosePointToPlane(
	const std::vector<Vector3f>& sourcePoints,
	const std::vector<Vector3f>& targetPoints,
	const std::vector<Vector3f>& targetNormals){

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

		for (int j = 0; j < 3; j++)
		{
			A(i, j) = nxs[j];
			A(i, j + 3) = n[j];
		}

		b(i) = n.dot(d) - n.dot(s);

		unsigned int j = i + 1;

		// Coordinate x
		A(j, 0) = 0;
		A(j, 1) = s[2];
		A(j, 2) = -s[1];
		A(j, 3) = 1;
		A(j, 4) = 0;
		A(j, 5) = 0;

		b(j) = d[0] - s[0];
		j++;

		// Coordinate y
		A(j, 0) = -s[2];
		A(j, 1) = 0;
		A(j, 2) = s[0];
		A(j, 3) = 0;
		A(j, 4) = 1;
		A(j, 5) = 0;

		b(j) = d[1] - s[1];
		j++;

		// Coordinate z
		A(j, 0) = s[1];
		A(j, 1) = -s[0];
		A(j, 2) = 0;
		A(j, 3) = 0;
		A(j, 4) = 0;
		A(j, 5) = 1;

		b(j) = d[2] - s[2];
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
