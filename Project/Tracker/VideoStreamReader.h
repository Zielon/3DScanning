#pragma once


class VideoStreamReader
{
public:

	VideoStreamReader() : m_height_rgb(0), m_width_rgb(0), m_height_depth(0), m_width_depth(0), newFrameIndex(0), lastFrameReadIndex(0)
	{

	}
	
	virtual ~VideoStreamReader()
	{
	}

	virtual bool startReading() = 0; 
	virtual bool stopReading() = 0;

	/**
	* returns the next frame for the programm to process
	* will block, if no new frames are available
	*
	* rgb: stores rgb frame
	* depth: stores depth frame
	* skip: true -> get the most recent frame
	*		false-> get the next frame in sequence
	*		if the programm is not realtime capable it might have to skip frames in order to keep up with the users movement
	*		skipping frames, however, will have negative influences in accuracy and skipping a too large window will cause registration to fail 
	*
	*/
	void getNextFrame(float** rgb, float** depth, bool skip = true)
	{
		waitForNextFrame(); 
		++newFrameIndex; 
		if (skip)
		{
			lastFrameReadIndex = newFrameIndex; 
			getLatestFrame(rgb, depth); 
		}
		else
		{
			++lastFrameReadIndex; 
			getSequentialFrame(rgb, depth);
		}
	}

	/**
	* tries returns the next frame for the programm to process
	* will return immideatly if no new frames available
	*
	* rgb: stores rgb frame
	* depth: stores depth frame
	* skip: true -> get the most recent frame
	*		false-> get the next frame in sequence
	*		if the programm is not realtime capable it might have to skip frames in order to keep up with the users movement
	*		skipping frames, however, will have negative influences in accuracy and skipping a too large window will cause registration to fail
	*
	* returns: true if frame was read; false, if no frame available
	*/
	bool tryGetNextFrame(float** rgb, float** depth, bool skip = true)
	{
		if (lastFrameReadIndex == newFrameIndex && !nextFrameAvailable()) // no new frames
		{
			return false;
		}
		getNextFrame(rgb, depth, skip);
		return true;

	}
	int m_height_rgb, m_width_rgb, m_height_depth, m_width_depth;


	unsigned long getCurrentFrameIndex() { return newFrameIndex; }

protected:

	//FIXME: bad naive impl
	virtual void waitForNextFrame()
	{
		while (lastFrameReadIndex == newFrameIndex && !nextFrameAvailable());
	}


	virtual bool nextFrameAvailable() = 0; 

	virtual int getSequentialFrame(float** rgb, float** depth) = 0;
	virtual int getLatestFrame(float** rgb, float** depth) = 0;



private: 
	unsigned long lastFrameReadIndex;
	unsigned long newFrameIndex;

};