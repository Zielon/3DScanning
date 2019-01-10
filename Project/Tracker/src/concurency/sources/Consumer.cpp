#include "../headers/Consumer.h"

template <class T>
void Consumer<T>::run(std::function<void(T)> func){
	while (true)
	{
		if (!m_running) break;

		T element = m_buffer->remove();
		func(element);
	}
}

template <class T>
void Consumer<T>::stop(){
	m_running = false;
}
