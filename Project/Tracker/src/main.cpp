#include "tests/windows/WindowsTests.h"
#include "tests/testBase.h"

int main(int argc, char** argv){

	//auto windows = new WindowsTests();
	auto test = new testBase();
	test->run();

	//windows->run();

	//SAFE_DELETE(windows);
	SAFE_DELETE(test);

	return 0;
}
