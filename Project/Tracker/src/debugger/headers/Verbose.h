#ifndef TRACKER_VERBOSE_H
#define TRACKER_VERBOSE_H

#include <string>
#include <ctime>

class Verbose
{
public:
	static void start();

	static void stop(std::string message);

private:
	static clock_t begin;
	static clock_t end;
};

#endif
