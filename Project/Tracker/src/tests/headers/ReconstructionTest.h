#pragma once
#include "TestBase.h"

class ReconstructionTest :
	public TestBase
{
public:
	void run() override{
		//this->reconstructionTest();
		//this->reconstructionTestWithOurTracking();
<<<<<<< HEAD
		this->pointCloudTest();
=======
		this->reconstructionTestSensor();
		//this->pointCloudTest();
>>>>>>> master
		//this->pointCloudTestWithICP();
		//this->unityIntegrationTest(); 
	};

private:
	void reconstructionTest() const;
	void pointCloudTestWithICP() const;
	void reconstructionTestWithOurTracking() const;
	void reconstructionTestSensor() const;
	void pointCloudTest() const;
	void unityIntegrationTest() const; 
};
