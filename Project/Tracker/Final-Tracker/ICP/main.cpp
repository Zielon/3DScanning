#include "src/tests/headers/DataStreamTest.h"
#include "src/tests/headers/ReconstructionTest.h"
#include "src/tests/headers/TrackerTest.h"

int main(int argc, char** argv){

	auto tests = std::vector<TestBase*>();

	//tests.push_back(new TrackerTest());
	tests.push_back(new ReconstructionTest());
	//tests.push_back(new DataStreamTest());

	for (auto test : tests)
	{
		test->run();
		SAFE_DELETE(test);
	}

	return 0;
}
