#pragma once
#include "TestBase.h"

class TrackerTest :
	public TestBase
{
public:
	void run() override{
		//this->cameraPoseTest();
		this->processedMapsTest();
	};
private:
	void frameDistanceTest();
	void cameraPoseTest();
	void processedMapsTest();
};
