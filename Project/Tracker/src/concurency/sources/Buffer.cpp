#include "../headers/Buffer.h"

template <class T>
void Buffer<T>::add(T element){
	while (true)
	{
		std::unique_lock<std::mutex> locker(m_mutex);
		m_cond.wait(locker, [this]()
		{
			return m_buffer.size() < m_size;
		});
		m_buffer.push_back(element);
		locker.unlock();
		m_cond.notify_all();
		return;
	}
}

template <class T>
T Buffer<T>::remove(){
	while (true)
	{
		std::unique_lock<std::mutex> locker(m_mutex);
		m_cond.wait(locker, [this]()
		{
			return !m_buffer.empty();
		});
		T element = m_buffer.back();
		m_buffer.pop_back();
		locker.unlock();
		m_cond.notify_all();
		return element;
	}
}
