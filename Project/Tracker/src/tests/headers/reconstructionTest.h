#pragma once
#include "TestBase.h"

class ReconstructionTest :
	public TestBase
{
public:

	void run() override{
		this->reconstructionTest();
		//this->pointCloudTest();
		//rhis-> pointCloudWithIcpTest();
	};

private:
	void reconstructionTest() const;
	void pointCloudWithIcpTest();
	void pointCloudTest() const;
};
