#ifndef TRACKER_VERBOSE_H
#define TRACKER_VERBOSE_H

#include <string>
#include <ctime>
#include <windows.h>

enum Type
{
	INFO,
	WARNING,
	SUCCESS
};

class Verbose
{
public:
	static void start();

	static void stop(std::string message, Type type = INFO);

	static void message(std::string message, Type type = INFO);

private:
	static clock_t begin;
	static clock_t end;
	static HANDLE hConsole;
	static void changeConsole(Type type);
};

#endif
