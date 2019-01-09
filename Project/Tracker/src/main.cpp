#include "tests/general/GeneralTests.h"
#include "tests/windows/WindowsTests.h"

int main(int argc, char** argv){

	auto windows = new WindowsTests();
	auto general = new GeneralTests();

	windows->run();
	general->run();

	return 0;
}
