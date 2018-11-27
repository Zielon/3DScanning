/**
 * Course: 3D Scanning and Motion Capture
 * File: World.h
 * Purpose: Fourth exercise of the course.
 * @author Juan Raul Padron Griffe, Wojciech Zielonka
 * @version 1.0 27/11/2018
*/

#pragma once

#include <vector>
#include <algorithm>
#include <Eigen/Dense>
#include <iostream>
#include <climits>
#include <bitset>
#include <unordered_map>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#define N_FRAMES 32

#define IMAGE_WIDTH 640 
#define IMAGE_HEIGHT 480


struct CostFunctor {
	CostFunctor(int idx0_, int idx1_, Eigen::Vector2d kp0_, Eigen::Vector2d kp1_, float depth_, const double *pose0_, const Eigen::Matrix4d *K_) {
		idx0 = idx0_;
		idx1 = idx1_;
		kp0 = kp0_;
		kp1 = kp1_;
		depth = depth_;
		pose0 = pose0_;
		K = K_;
	}


	template <typename T>
	void inv(const T *pose, T *inv) const {
		T rinv[3] = { -pose[0], -pose[1], -pose[2] };
		T tinv[3]; ceres::AngleAxisRotatePoint(rinv, pose + 3, tinv); tinv[0] *= -1; tinv[1] *= -1; tinv[2] *= -1;
		inv[0] = rinv[0];
		inv[1] = rinv[1];
		inv[2] = rinv[2];
		inv[3] = tinv[0];
		inv[4] = tinv[1];
		inv[5] = tinv[2];
	};

	template <typename T>
	void apply_pose(const T *pose, T *p0, T *p) const {
		ceres::AngleAxisRotatePoint(pose, p0, p);
		p[0] += pose[3]; p[1] += pose[4]; p[2] += pose[5];
	};

	template <typename T>
	inline void set_from_real(T *array, const double *v, const int n) const {
		for (int i = 0; i < n; i++)
			array[i] = T(v[i]);
	}

	template <typename T>
	inline void set_from_T(T *array, const T *v, const int n) const {
		for (int i = 0; i < n; i++)
			array[i] = v[i];
	}



	template <typename T>
	bool operator()(const T* params, T* residual) const {
		// TODO: Task 3.2
		T fx = T((*K)(0, 0));
		T fy = T((*K)(1, 1));
		T cx = T((*K)(0, 2));
		T cy = T((*K)(1, 2));

		// -> camera poses
		T params0[6], params1[6];

		if (idx0 == 0)
			set_from_real(params0, pose0, 6);
		else
			set_from_T(params0, params + 6 * idx0, 6);

		if (idx1 == 0)
			set_from_real(params1, pose0, 6);
		else
			set_from_T(params1, params + 6 * idx1, 6);

		T params0_inv[6]; inv(params0, params0_inv);
		T params1_inv[6]; inv(params1, params1_inv);
		// <-

		// -> img0 to world
		T p0[3];
		// p0[0] = ...
		// p0[1] = ...
		// p0[2] = ...

		T pw[3];
		apply_pose(params0, p0, pw);
		// <-

		// -> world to img1
		T p1[3];
		// apply_pose(?, pw, p1);

		T pred[2];
		// pred[0] = ...
		// pred[1] = ...
		// <-

		// figure out dim (tip: residuals are in pixel space)
		// residual[0] = ...
		// residual[0 + 1] = ...
		// residual[...] = ...
		// residual[dim - 1] = ...

		return true;
		// <-
	}

	Eigen::Vector2d kp0, kp1;
	int idx0, idx1;
	float depth;
	const double* pose0;
	const Eigen::Matrix4d* K;

};


class World {
public:


	World() {
		keypoints.resize(N_FRAMES);
		descriptors.resize(N_FRAMES);
	}

	void set_data(std::vector<cv::Mat> &rgb_, std::vector<cv::Mat> &depth_, Eigen::MatrixXd &poses_, Eigen::Matrix4d &K_) {
		rgb = rgb_;
		depth = depth_;
		poses = poses_;
		K = K_;
	}

	void show_pics() {
		// -> TODO: Task 1.1

		//Color maps
		//cv::namedWindow( "Pictures", cv::WINDOW_AUTOSIZE );// Create a window for display.

		for (int i = 0; i < rgb.size(); i++){
			cv::imshow( "Color maps", rgb[i] );
			cv::waitKey();
		}

		//Depth maps (improve visualization)
		for (int i = 0; i < rgb.size(); i++){
			cv::imshow( "Depth maps", depth[i] );
			cv::waitKey();
		}

		// <-
	}

	void extract_features() {
		// -> TODO Task 1.2

		cv::Ptr< cv::FeatureDetector > detector = cv::ORB::create();//Create ORB Detector


		for (int i = 0; i < N_FRAMES; i++) {

			detector->detect(rgb[i], keypoints[i]);
		}
		// <-
	}

	void describe_features() {
		// -> TODO Task 1.3

		cv::Ptr< cv::DescriptorExtractor > descriptor = cv::ORB::create();//Create ORB Descriptor

		for (int i = 0; i < N_FRAMES; i++) {
			descriptor->compute ( rgb[i], keypoints[i], descriptors[i] );
		}

		// <-
	}

	int32_t hamming_distance(cv::Mat &d0, cv::Mat &d1) {
		int dist = 0;
		for (int i = 0; i < 32; i++) {
			dist += std::bitset<8>(d0.at<uint8_t >(0, i) ^ d1.at<uint8_t >(0, i)).count();
		}
		return dist;
	}

	void brute_force(int idx0, int idx1, std::vector<std::pair<int, int>> &matches) {
		// -> TODO Task 2.1

		matches.resize(descriptors[idx0].rows, { -1, INT_MAX });

		//std::cout << descriptors[idx0] << std::endl << "***************" << std::endl;

		for (int i = 0; i < descriptors[idx0].rows; i++) {

			cv::Mat d0 = descriptors[idx0].row(i);
			int32_t dist = 0;

			//std::cout << d0 << std::endl;

			for (int j = 0; j < descriptors[idx1].rows; j++) {

				cv::Mat d1 = descriptors[idx1].row(j);

				dist = this->hamming_distance(d0, d1);

				if (dist < matches[i].second){

					matches[i].first = j;
					matches[i].second = dist;
				}
			}
		}
		// <-
	}

	void filter_matches(int idx0, int idx1, std::vector<std::pair<int, int>> &matches, std::vector<cv::DMatch> &filtered) {
		// -> TODO Task 2.2

		for (int i = 0; i < (int)matches.size(); i++) {

			//The Hamming distance should be lower than 40
			if (matches[i].second >= 40) continue;


			//The L2-distance between keypoints should be lower than 40 pixels

			cv::Point2f p1 = keypoints[idx0][i].pt;
			cv::Point2f p2 = keypoints[idx1][matches[i].first].pt;

			double dist = cv::norm(p1 - p2);

			if ( (int)dist >= 40) continue;

			filtered.emplace_back(cv::DMatch(i, matches[i].first, matches[i].second));
		}

		// <-
	}


	inline uint32_t make_key(uint32_t i, uint32_t j) {
		return (uint32_t)i << 0 | (uint32_t)j << 16;
	};

	int32_t clamp(int32_t n, int32_t lower, int32_t upper) {
		return std::max(lower, std::min(n, upper));
	}

	void match_features() {
		for (uint32_t i = 0; i < N_FRAMES; i++) {
			uint32_t start = clamp(i - 3, 0, N_FRAMES - 1);
			uint32_t end = clamp(i + 3, 0, N_FRAMES - 1);
			for (uint32_t j = start; j < end; j++) {
				if (i == j)
					continue;
				std::vector<std::pair<int, int>> matches_all;
				brute_force(i, j, matches_all);

				std::vector<cv::DMatch> matches_filtered;
				filter_matches(i, j, matches_all, matches_filtered);
				matches[make_key(i, j)] = matches_filtered;
			}
		}
	}

	void show_matches() {
		cv::Mat img_matches;
		cv::drawMatches(rgb[0], keypoints[0], rgb[2], keypoints[2], matches[make_key(0, 2)], img_matches);
		cv::imshow("Good Matches", img_matches);
		cv::waitKey();
	}

	void bootstrap() {
		params.resize(6, N_FRAMES);
		params.colwise() = poses.col(0);
	}

	void construct_cost_function() {
		// -> TODO: Task 3.1
		bootstrap();
		int counter = 0;
		for (int i = 0; i < N_FRAMES; i++) {
			for (int j = 0; j < N_FRAMES; j++) {
				std::vector<cv::DMatch> matches_filtered; // = ...
				for (auto & m : matches_filtered) {
					cv::Point2i kp0; // = ...
					cv::Point2i kp1; // = ...
					float d = depth[i].at<float>(kp0.y, kp0.x);
					if (d == 0)
						continue;
					// CostFunctor *ref = new CostFunctor(?, ?, {?, ?}, {?, ?}, d, poses.col(0).data(), &K);
					// ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<CostFunctor, ?, 6*?>(ref);
					// problem.AddResidualBlock(cost_function, NULL, params.data());
					counter++;
				}
			}
		}
		std::cout << "n-residuals: " << counter << std::endl;
		// <-
	}

	void solve() {
		// ->  cost before optimization
		double cost_final = 0;
		problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost_final, NULL, NULL, NULL);
		std::cout << "cost-before: " << cost_final << std::endl;
		// <-

		// -> solve
		ceres::Solver::Options options;
		ceres::Solver::Summary summary;
		ceres::Solve(options, &problem, &summary);
		problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost_final, NULL, NULL, NULL);
		std::cout << "cost-after: " << cost_final << std::endl;
		// <-
	}

	void evaluate() {
		//Eigen::Vector3d error_trans(0, 0, 0);
		double error_trans = 0;
		for (int i = 1; i < N_FRAMES; i++) {
			error_trans += (params.col(i).segment(3, 3) - poses.col(i).segment(3, 3)).norm();
			//std::cout << "pred: " << params.col(i).segment(3, 3).transpose() << " gt: "  << poses.col(i).segment(3, 3).transpose() << std::endl;
		}
		error_trans /= N_FRAMES;

		std::cout << "Average error [in meter] from GT: " << error_trans << std::endl; // <-- should be lower than 0.1m
		std::cout << "Accumulated drift [in meter] at last frame: " << (params.col(N_FRAMES - 1).segment(3, 3) - poses.col(N_FRAMES - 1).segment(3, 3)).norm() << std::endl;
	}


public:
	std::vector<cv::Mat> rgb; // <-- rgb frames
	std::vector<cv::Mat> depth; // <-- depth maps
	Eigen::MatrixXd poses; // <-- camera poses
	Eigen::Matrix4d K; // <-- intrinsic matrix

	Eigen::MatrixXd params; // <-- pose params (to be optimized over)
	ceres::Problem problem; // <-- ceres problem struct

	std::vector<std::vector<cv::KeyPoint>> keypoints;
	std::vector<cv::Mat> descriptors;
	std::unordered_map<uint32_t, std::vector<cv::DMatch>> matches;
};
