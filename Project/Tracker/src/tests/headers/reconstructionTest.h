#pragma once
#include "TestBase.h"

class ReconstructionTest :
	public TestBase
{
public:
	void run() override{
		//this->meshTest();
		//this->reconstructionTest();
		this->reconstructionTestWithOurTracking();
		//this->pointCloudTest();
	};

private:
	void reconstructionTest() const;
	void reconstructionTestWithOurTracking() const;

	void pointCloudTest() const;
};
