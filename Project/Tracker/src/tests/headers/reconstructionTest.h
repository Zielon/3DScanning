#pragma once
#include "TestBase.h"

class ReconstructionTest :
	public TestBase
{
public:

	void run() override{
		//this->meshTest();
		//this->reconstructionTest();
		this->pointCloudTest();
	};

private:
	void reconstructionTest() const;
	void pointCloudWithIcpTest();
	void pointCloudTest() const;
};
