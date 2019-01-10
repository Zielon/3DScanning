#include "../headers/Verbose.h"
#include <iostream>

clock_t Verbose::begin = clock();
clock_t Verbose::end = clock();

void Verbose::start(){
	begin = clock();
}

void Verbose::stop(std::string message = ""){
	end = clock();
	const double seconds = double(end - begin) / CLOCKS_PER_SEC;
	std::cout << message << " [ " << seconds << " ] " << std::endl;
}
