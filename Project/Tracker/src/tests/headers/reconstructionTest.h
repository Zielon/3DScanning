#pragma once
#include "TestBase.h"

class ReconstructionTest :
	public TestBase
{
public:
	void run() override{
		//this->meshTest();
		//this->reconstructionTest();
		//this->pointCloudTest();
		this->pointCloudTestWithICP();
	};

private:
	void reconstructionTest() const;
	void pointCloudTestWithICP() const;
	void pointCloudTest() const;
};
