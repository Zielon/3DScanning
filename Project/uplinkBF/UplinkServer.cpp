#define WIN32_LEAN_AND_MEAN

#include <iostream>




#include "StructureSensor.h"


using namespace std; 


int main(int argc, char** argv)
{
	StructureSensor sensor; 

	sensor.startReceivingFrames(); 
	if (FAILED(sensor.CreateFirstConnected()))
	{
		std::cout << "CreateFirstConnected failed"<< std::endl;
	}

	if (FAILED(sensor.ProcessNextFrame()))
	{
		std::cout << "ProcessNextFrame failed"<< std::endl;
	}


}