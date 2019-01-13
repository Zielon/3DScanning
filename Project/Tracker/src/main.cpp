#define USE_SENSOR 1
#define USE_RECONSTRUCTION 0

#if USE_RECONSTRUCTION 

#include "tests/windows/WindowsTests.h"

#endif

#if USE_SENSOR 

#include "tests/windows/VideoStreamTests.h"

#endif

int main(int argc, char** argv){

	#if USE_RECONSTRUCTION 
		auto windows = new WindowsTests();

		windows->run();

		SAFE_DELETE(windows);

	#endif

	#if USE_SENSOR 
		auto sensorTests = new VideoStreamTests();

		sensorTests->run();

		SAFE_DELETE(sensorTests);
	#endif

	return 0;
}
