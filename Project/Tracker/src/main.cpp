#include "tests/testBase.h"

int main(int argc, char** argv){

	auto test = new testBase();
	test->run();

	SAFE_DELETE(test);

	return 0;
}
