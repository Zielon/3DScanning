#ifndef PROGRESSBAR_PROGRESSBAR_HPP
#define PROGRESSBAR_PROGRESSBAR_HPP

#include <chrono>
#include <iostream>

class ProgressBar
{
private:
	unsigned int m_ticks = 0;

	const unsigned int m_total_ticks;
	const unsigned int m_bar_width;
	const char m_complete_char = '=';
	const char m_incomplete_char = ' ';
	const std::chrono::steady_clock::time_point m_start_time = std::chrono::steady_clock::now();

public:
	ProgressBar(unsigned int total, unsigned int width, char complete, char incomplete) :
		m_total_ticks{total}, m_bar_width{width}, m_complete_char{complete}, m_incomplete_char{incomplete}{}

	ProgressBar(unsigned int total, unsigned int width) : m_total_ticks{total}, m_bar_width{width}{}

	unsigned int operator++(){
		return ++m_ticks;
	}

	void set(int tick){
		m_ticks = tick;
	}

	void display() const{
		float progress = (float)m_ticks / m_total_ticks;
		int pos = (int)(m_bar_width * progress);

		std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
		auto time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - m_start_time).count();

		std::cout << "[";

		for (int i = 0; i < m_bar_width; ++i)
		{
			if (i < pos) std::cout << m_complete_char;
			else if (i == pos) std::cout << ">";
			else std::cout << m_incomplete_char;
		}
		std::cout << "] " << int(progress * 100.0) << "% "
			<< float(time_elapsed) / 1000.0 << "s\r";
		std::cout.flush();
	}

	void done() const{
		display();
		std::cout << std::endl;
	}
};

#endif //PROGRESSBAR_PROGRESSBAR_HPP
