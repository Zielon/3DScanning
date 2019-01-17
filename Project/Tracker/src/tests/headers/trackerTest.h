#include "TestBase.h"

class TrackerTest :
	public TestBase
{
public:
	void run() {
		//this->cameraPoseTest();
		this->frameDistanceTest();
	};
	void  cameraPoseTest();
	void  frameDistanceTest();
};

