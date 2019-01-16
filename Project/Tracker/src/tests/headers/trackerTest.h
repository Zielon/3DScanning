#pragma once
#include "TestBase.h"

class TrackerTest :
	public TestBase
{
public:
	void run() override{
		this->cameraPoseTest();
	};
private:

	void cameraPoseTest();
};
