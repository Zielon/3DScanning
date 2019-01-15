#include "TestBase.h"

class TrackerTest :
	public TestBase
{
public:
	void run() {
		this->cameraPoseTest();
	};
	void cameraPoseTest();
};

