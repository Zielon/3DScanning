#ifndef TRACKER_LIB_FUSION_H
#define TRACKER_LIB_FUSION_H

#include "../../Eigen.h"
#include "CameraParameters.h"
#include "vector"
#include "Voxel.h"
#include "PointCloud.h"

using namespace std;

/**
 * Volumetric fusion class
 */
class Fusion final
{
public:
	Fusion(CameraParameters camera_parameters);

	~Fusion();

	void integrate(const PointCloud& cloud, Matrix4f& pose);

	vector<vector<vector<Voxel*>>>& getTSDF();

private:
	void forAll(function<void(Voxel*, Vector3f)> func);

	Voxel* get(int i, int j, int k);

	CameraParameters m_camera_parameters;
	vector<vector<vector<Voxel*>>> m_voxles_space;
	int m_size = 2000;
};

#endif //TRACKER_LIB_FUSION_H
