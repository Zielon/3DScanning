#include "../headers/Fusion.h"

void Fusion::integrate(const std::vector<Vector3f> &cloud, Matrix4f &pose) {

	auto worldToCamera = pose.inverse();

}
