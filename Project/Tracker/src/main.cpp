#include "tests/headers/dataStreamTest.h"
#include "tests/headers/reconstructionTest.h"
#include "tests/headers/trackerTest.h"

int main(int argc, char** argv){

	auto track_test = new trackerTest();
	auto reconstruction_test = new reconstructionTest();
	auto data_stream_test = new dataStreamTest();
	//track_test->cameraPoseTest();
	reconstruction_test->pointCloudTest();
	//reconstruction_test->meshTest();
	//reconstruction_test->reconstructTest();
	//data_stream_test->vidReadTest();

	SAFE_DELETE(track_test);
	SAFE_DELETE(reconstruction_test);
	SAFE_DELETE(data_stream_test);
	return 0;
}
