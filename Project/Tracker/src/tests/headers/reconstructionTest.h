#include "TestBase.h"

class ReconstructionTest :
	public TestBase
{
public:
	void run() {
		this->meshTest();
		this->reconstructTest();
		this->pointCloudTest();
	};
	void meshTest();
	void reconstructTest();
	void pointCloudTest();
};