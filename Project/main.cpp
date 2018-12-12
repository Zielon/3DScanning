#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include "Tracker.h"

int main() {

	Tracker tracker;

	tracker.computerCameraPose(nullptr, nullptr, 0, 0);

	return 0;
}