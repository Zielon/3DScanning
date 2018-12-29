#include "XTionStreamReader.h"

#include <iostream>
#include <fstream>

int main(){

	XtionStreamReader *streamReader = new XtionStreamReader(false);

	std::cout << "Stream created properly" << std::endl;

	if (!streamReader->startReading()) {
		std::cout << "Failed to read input stream" << std::endl;
		return -1;
	}

	std::cout << "The reading process has started" << std::endl;

	std::cin.get();

	delete streamReader;

	return 0;
}