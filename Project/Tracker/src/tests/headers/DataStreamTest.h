#pragma once
#include "TestBase.h"
#include "../../data-stream/headers/Xtion2StreamReader.h"

#include <conio.h>

class DataStreamTest :
	public TestBase
{
public:
	void run() override{
		//this->vidReadTest();
		this->sensorTest(true);
	};

	int wasKeyboardHit(){ return (int)_kbhit(); }

	void vidReadTest();
	int sensorTest(bool useOpenni2);
};
