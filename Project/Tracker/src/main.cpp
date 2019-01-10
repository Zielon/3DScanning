#include "tests/windows/WindowsTests.h"

int main(int argc, char** argv){

	auto windows = new WindowsTests();

	windows->run();

	SAFE_DELETE(windows);

	return 0;
}
