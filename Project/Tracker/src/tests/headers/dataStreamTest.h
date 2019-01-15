#pragma once
#include "TestBase.h"

class DataStreamTest :
	public TestBase
{
public:
	void run() {
		this->vidReadTest();
	};
	void vidReadTest();
};

