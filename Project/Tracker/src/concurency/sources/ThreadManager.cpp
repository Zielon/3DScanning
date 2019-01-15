#include "../headers/ThreadManager.h"
#include <sstream>

std::vector<std::thread> ThreadManager::m_threads;

/// The thread list is static. You are adding to the global pool!
void ThreadManager::add(std::function<void()> func){
	m_threads.emplace_back([func](){
		func();
	});
}

/// You are waiting for the all threads from the global pool!
void ThreadManager::waitForAll(){
	std::for_each(m_threads.begin(), m_threads.end(), std::mem_fn(&std::thread::join));
	m_threads.clear();
}

void ThreadManager::waitForAll(std::vector<std::thread>& threads){
	std::for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));
	threads.clear();
}

std::string ThreadManager::getId(){
	auto id = std::this_thread::get_id();
	std::stringstream ss;
	ss << id;
	return ss.str();
}
