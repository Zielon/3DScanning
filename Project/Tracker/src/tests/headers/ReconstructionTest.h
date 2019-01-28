#pragma once
#include "TestBase.h"

class ReconstructionTest :
	public TestBase
{
public:
	void run() override{
		//this->reconstructionTest(1, 1);
		//this->reconstructionTestWithOurTracking(1);
		this->reconstructionTestSensor(100);
		//this->pointCloudTest();
		//this->pointCloudTestWithICP();
		//this->unityIntegrationTest(); 
	};

private:
	void reconstructionTest(int skip, int subsampling) const;

	void pointCloudTestWithICP() const;

	void reconstructionTestWithOurTracking(int increment) const;

	void reconstructionTestSensor(int mesh_index) const;

	void pointCloudTest() const;

	void unityIntegrationTest() const;
};