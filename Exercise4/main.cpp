#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>

#include "FreeImageHelper.h"

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include "World.h"

#ifndef N_FRAMES
#define N_FRAMES 8
#endif

//const std::string DATA_DIR = "assets/rgbd_dataset_freiburg3_teddy/";

void load_data(std::string filename, std::vector<cv::Mat> &rgb, std::vector<cv::Mat> &depth, Eigen::MatrixXd &poses, int skip, int n_frames) {
	assert(skip >= 1);
	std::string line;
	std::ifstream file(filename);
	int counter = 0, counter_tmp = 0;
	Eigen::Vector3d t0(0, 0, 0);
	Eigen::Quaterniond q0(1, 0, 0, 0);
	if (file.is_open()) {
		while (std::getline(file, line) && counter < n_frames) {
			if (counter_tmp++ % skip == 0) {
				std::vector<std::string> items;
				std::string item;
				std::stringstream ss(line);
				while (std::getline(ss, item, ' '))
					items.push_back(item);

				// -> load rgb
				std::string filename_rgb = std::string(DATA_DIR) + items[9];
				rgb[counter] = cv::imread(filename_rgb);
				// <-

				// -> load depth
				std::string filename_depth = std::string(DATA_DIR) + items[11];
				//cv::Mat depth1 = cv::imread(filename_depth);
				//depth1.convertTo(depth[counter], CV_32FC1, 1.0/5000.0);
				depth[counter] = cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32FC1);
				//std::vector<uint8_t> depth1(2 * IMAGE_WIDTH*IMAGE_HEIGHT, 0);

				FreeImageU16F dImage;
				dImage.LoadImageFromFile(filename_depth, IMAGE_WIDTH, IMAGE_HEIGHT);

				std::transform(dImage.data, dImage.data + dImage.w * dImage.h, depth[counter].ptr<float>(), [](float a) -> float {return a / 5000.0f; });
				// <-

				// -> pose
				Eigen::Vector3d t(std::stod(items[1]), std::stod(items[2]), std::stod(items[3])); // <- translation
				Eigen::Quaterniond q(std::stod(items[7]), std::stod(items[4]), std::stod(items[5]), std::stod(items[6])); // <- rotation in Eigen: w,x,y,z
				// -> making first timestamp as identity
				if (counter == 0) {
					t0 = t;
					q0 = q;
				}
				//t = t - t0;
				//q = q0.inverse()*q;
				//std::cout << "quat: " << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << std::endl;
				// <-
				Eigen::Vector3d a = Eigen::AngleAxisd(q).angle()*Eigen::AngleAxisd(q).axis(); // <-- convert to axis angle (because we will optimize over it)
				poses.col(counter) << a(0), a(1), a(2), t(0), t(1), t(2); // <- fill pose
				//std::cout << "**** " <<  poses.col(counter).transpose() << std::endl;

				// <-
				counter++;
			}
		}
		file.close();
	}
	assert(counter == n_frames);

};


int main() {
	// -> init containers and load data
	Eigen::MatrixXd poses(6, N_FRAMES);
	std::vector<cv::Mat> rgb(N_FRAMES);
	std::vector<cv::Mat> depth(N_FRAMES);
	load_data(std::string(DATA_DIR) + "/associated.txt", rgb, depth, poses, 4, N_FRAMES);
	float fx = 535.4, fy = 539.2, cx = 320.1, cy = 247.6; // <-- hard-coded camera intrinsics
	Eigen::Matrix4d K = Eigen::Matrix4d::Identity(); K(0, 0) = fx; K(1, 1) = fy; K(0, 2) = cx; K(1, 2) = cy;
	// <-


	// -> create workspace. let's call it world
	World world;
	world.set_data(rgb, depth, poses, K);
	std::cout << "Loading data done..." << std::endl;
	// <-

	// -> show images
	world.show_pics();
	// <-

	// -> extract keypoints
	world.extract_features();
	std::cout << "Feature extraction done..." << std::endl;
	// <-

	// -> describe keypoints
	world.describe_features();
	std::cout << "Feature description done..." << std::endl;
	// <-

	// -> match features
	world.match_features();
	std::cout << "Feature matching done..." << std::endl;
	// <-

	// -> show matches
	world.show_matches();
	// <-

	// -> construct cost function via Ceres
	world.construct_cost_function();
	std::cout << "Constructing cost function done..." << std::endl;
	// <-

	// -> solve BA
	world.solve();
	std::cout << "Solve done..." << std::endl;
	// <-

	// -> evaluate BA
	world.evaluate();
	std::cout << "Evaluation done..." << std::endl;
	// <-

	return 0;
}