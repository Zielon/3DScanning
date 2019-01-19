#pragma once
#include "TestBase.h"

class ReconstructionTest :
	public TestBase
{
public:
	void run() override{
		//this->reconstructionTest();
		//this->reconstructionTestWithOurTracking();
		//this->pointCloudTest();
		this->pointCloudTestWithICP();
	};

private:
	void reconstructionTest() const;
	void pointCloudTestWithICP() const;
	void reconstructionTestWithOurTracking() const;
	void pointCloudTest() const;
};
