#include "tests/headers/DataStreamTest.h"
#include "tests/headers/ReconstructionTest.h"
#include "tests/headers/TrackerTest.h"

int main(int argc, char** argv){

	auto tests = std::vector<TestBase*>();

	tests.push_back(new TrackerTest());
	tests.push_back(new ReconstructionTest());
	tests.push_back(new DataStreamTest());

	for (auto test : tests) {
		test->run();
	}

	for (auto test : tests) {
		SAFE_DELETE(test);
	}
	return 0;
}
