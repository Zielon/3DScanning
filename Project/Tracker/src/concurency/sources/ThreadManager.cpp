#include "../headers/ThreadManager.h"

std::vector<std::thread> ThreadManager::threads;

/// The thread list is static. You are adding to the global pool!
void ThreadManager::add(std::function<void()> func){
	threads.emplace_back([func]()
	{
		func();
	});
}

/// You are waiting for the all threads from the global pool!
void ThreadManager::waitForAll(){
	std::for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));
	threads.clear();
}

void ThreadManager::waitForAll(std::vector<std::thread>& vector){
	std::for_each(vector.begin(), vector.end(), std::mem_fn(&std::thread::join));
	vector.clear();
}
