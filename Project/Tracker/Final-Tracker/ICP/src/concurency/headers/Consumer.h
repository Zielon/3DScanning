#ifndef TRACKER_LIB_CONSUMER_H
#define TRACKER_LIB_CONSUMER_H
#include "Buffer.h"

template <class T>
class Consumer final
{
public:
	Consumer(Buffer<T>* buffer): m_buffer(buffer){}

	void run(std::function<void(T)> func);

	void stop();

private:
	Buffer<T>* m_buffer;
	bool m_running = true;
};

#endif
