#include "../headers/Producer.h"

template <class T>
void Producer<T>::run(std::function<void(T)> func){
	while (true)
	{
		m_buffer->add(nullptr);
	}
}
