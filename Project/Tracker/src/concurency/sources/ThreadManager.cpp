#include "../headers/ThreadManager.h"

std::vector<std::thread> ThreadManager::threads;

void ThreadManager::add(std::function<void()> func) {
	threads.emplace_back([func]()
	{
		func();
	});
}

void ThreadManager::waitForAll() {
	std::for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));
	threads.clear();
}
