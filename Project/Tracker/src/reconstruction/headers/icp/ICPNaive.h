#pragma once
#ifndef PROJECT_ICP_NAIVE_H
#define PROJECT_ICP_NAIVE_H

#include "ICP.h"
#include "../../sources/NearestNeighbor.hpp"
#include "../../sources/ProcrustesAligner.hpp"

class ICPNaive final : public ICP
{
public:
	ICPNaive();

	~ICPNaive();

	Matrix4f estimatePose(std::shared_ptr<PointCloud> source, std::shared_ptr<PointCloud> target) override;

private:
	NearestNeighborSearch* m_nearestNeighbor;
	ProcrustesAligner* m_procrustesAligner;

	#if _DEBUG
	int m_number_iterations = 1;
	#else
	int m_number_iterations = 10;
	#endif

	std::vector<Vector3f> transformPoints(const std::vector<Vector3f>& sourcePoints, const Matrix4f& pose) const;

	std::vector<Vector3f> transformNormals(const std::vector<Vector3f>& sourceNormals, const Matrix4f& pose);

	Matrix4f estimatePosePointToPoint(
		const std::vector<Vector3f>& sourcePoints,
		const std::vector<Vector3f>& targetPoints) const;

	Matrix4f estimatePosePointToPlane(
		const std::vector<Vector3f>& sourcePoints,
		const std::vector<Vector3f>& targetPoints,
		const std::vector<Vector3f>& targetNormals) const;

	void pruneCorrespondences(
		const std::vector<Vector3f>& sourceNormals,
		const std::vector<Vector3f>& targetNormals, std::vector<Match>& matches);
};

#endif //PROJECT_ICP_NAIVE_H
