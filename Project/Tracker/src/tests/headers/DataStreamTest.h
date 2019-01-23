#pragma once
#include "TestBase.h"
#include "../../data-stream/headers/Xtion2StreamReader.h"

class DataStreamTest :
	public TestBase
{
public:
	void run() override{
		//this->vidReadTest();
		this->sensorTest(true);
	};

	void vidReadTest();
	int sensorTest(bool useOpenni2);
};
