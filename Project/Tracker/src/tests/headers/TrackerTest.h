#pragma once
#include "TestBase.h"

class TrackerTest :
	public TestBase
{
public:
	void run() override{
		this->cameraPoseTest(1, 1);
		//this->processedMapsTest();
	};
private:

	void cameraPoseTest(int skip, int subsampling);

	void processedMapsTest();
};
