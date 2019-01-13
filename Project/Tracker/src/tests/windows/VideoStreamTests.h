#ifndef VIDEO_STREAM_TESTS_H
#define VIDEO_STREAM_TESTS_H

#include "../../data-stream/headers/DatasetVideoStreamReader.h"
#include "../../data-stream/headers/XtionStreamReader.h"
#include "../../data-stream/headers/Xtion2StreamReader.h"

#include <conio.h>

class VideoStreamTests {
public:
	void run();

private:

	int wasKeyboardHit();
	int sensorTest(bool useOpenni2);
};

#endif //VIDEO_STREAM_TESTS_H
