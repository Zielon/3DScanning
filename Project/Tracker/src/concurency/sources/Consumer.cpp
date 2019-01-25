#include "../headers/Consumer.h"
#include "../../debugger/headers/Verbose.h"
#include "../headers/ThreadManager.h"

template <class T>
void Consumer<T>::run(std::function<void(T)> func){
	while (true)
	{
		if (!m_running)
			break;

		T element = m_buffer->remove();

		if (!element)
			continue;

		func(element);
	}

	Verbose::message("Consumer has been terminated -> " + ThreadManager::getId());
}

template <class T>
void Consumer<T>::stop(){
	m_running = false;
}
