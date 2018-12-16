#include <iostream>

#include "Tracker.h"

extern "C" __declspec(dllexport) int test() {
	return 8;
}

int main() {

    Tracker tracker;

    tracker.computerCameraPose(nullptr, nullptr, 0, 0);

    return 0;
}