#ifndef TRACKER_LIB_THREAD_MANAGER_H
#define TRACKER_LIB_THREAD_MANAGER_H

#include <thread>
#include <vector>
#include <algorithm>

class ThreadManager
{
public:
	static void add(std::function<void()> func);

	static void waitForAll();

	static void waitForAll(std::vector<std::thread>& threads);

	static std::vector<std::thread> m_threads;
};

#endif
