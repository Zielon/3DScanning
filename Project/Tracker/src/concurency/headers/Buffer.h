#ifndef TRACKER_LIB_BUFFER_H
#define TRACKER_LIB_BUFFER_H

#include <deque>
#include <mutex>

template <class T>
class Buffer
{
public:

	void add(T element);

	T remove();

	bool isEmpty();

private:
	std::mutex m_mutex;
	std::condition_variable m_cond;
	std::deque<T> m_buffer;
	const unsigned int m_size = 100;
};

#endif
