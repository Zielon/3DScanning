#pragma once
#include "TestBase.h"

class DataStreamTest :
	public TestBase
{
public:
	void run() override{
		this->vidReadTest();
	};

	void vidReadTest();
};
