#ifndef TRACKER_LIB_PRODUCER_H
#define TRACKER_LIB_PRODUCER_H
#include "Buffer.h"

template <class T>
class Producer final
{
public:
	Producer(Buffer<T>* buffer) : m_buffer(buffer){}

	void run(std::function<void(T)> func);

private:
	Buffer<T>* m_buffer;
};

#endif
