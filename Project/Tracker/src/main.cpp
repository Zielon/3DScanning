#include "tests/headers/DataStreamTest.h"
#include "tests/headers/ReconstructionTest.h"
#include "tests/headers/TrackerTest.h"

int main(int argc, char** argv){

	auto track_test = new TrackerTest();
	auto reconstruction_test = new ReconstructionTest();
	auto data_stream_test = new DataStreamTest();
	track_test->cameraPoseTest();
	//reconstruction_test->pointCloudTest();
	//reconstruction_test->meshTest();
	//reconstruction_test->reconstructTest();
	//data_stream_test->vidReadTest();

	SAFE_DELETE(track_test);
	SAFE_DELETE(reconstruction_test);
	SAFE_DELETE(data_stream_test);
	return 0;
}
