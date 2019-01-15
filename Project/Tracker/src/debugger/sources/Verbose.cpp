#include "../headers/Verbose.h"
#include <iostream>

clock_t Verbose::begin = clock();
clock_t Verbose::end = clock();

HANDLE Verbose::hConsole = GetStdHandle(STD_OUTPUT_HANDLE);

void Verbose::start(){
	begin = clock();
}

void Verbose::stop(std::string message = "", Type type){
	end = clock();
	const double seconds = double(end - begin) / CLOCKS_PER_SEC;
	changeConsole(type);
	#ifdef TESTING
	std::cout << message << " [ " << seconds << " s ] " << std::endl;
	#endif
}

void Verbose::message(std::string message, Type type){
	changeConsole(type);
	#ifdef TESTING
	std::cout << message << std::endl;
	#endif
}

void Verbose::changeConsole(Type type){
	switch (type)
	{
	case SUCCESS:
		SetConsoleTextAttribute(hConsole, 10);
		break;
	case INFO:
		SetConsoleTextAttribute(hConsole, 11);
		break;
	case WARNING:
		SetConsoleTextAttribute(hConsole, 12);
		break;
	default:
		SetConsoleTextAttribute(hConsole, 15);
	}
}
