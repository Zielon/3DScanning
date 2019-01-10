#include "../headers/Consumer.h"

template <class T>
void Consumer<T>::run(std::function<void(T)> func){
	while (true)
	{
		T element = m_buffer->remove();
		func(element);
	}
}
