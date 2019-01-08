#include "stdafx.h"
#include "CppUnitTest.h"

#include "../../src/TrackerContext.h"
#include "../../src/reconstruction/headers/Tracker.h"
#include "../../src/data-stream/headers/DatasetVideoStreamReader.h"
#include "../../src/ExportDLL.h"
#include <direct.h>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

const std::string DATASET_DIR = "\\..\\..\\..\\MarkerlessAR_Unity\\Datasets\\freiburg\\";

namespace Tests
{		
	TEST_CLASS(UnitTest1)
	{
	public:
		
		TEST_METHOD(dllVidReadTest)
		{
			// TODO: Your test code here

			std::cout << "START dllVidReadTest()" << std::endl;

			char cCurrentPath[FILENAME_MAX];

			_getcwd(cCurrentPath, sizeof(cCurrentPath));

			strcpy(cCurrentPath + strlen(cCurrentPath), DATASET_DIR.c_str());

			TrackerContext *pc = static_cast<TrackerContext*>(createContext(cCurrentPath));

			byte *img = new byte[getImageWidth(pc) * getImageHeight(pc) * 3];

			float pose[16];

			for (int i = 0; i < 3000; ++i)
			{
				dllMain(pc, img, pose);

				cv::Mat dllmat = cv::Mat(getImageHeight(pc), getImageWidth(pc), CV_8UC3, img);
				cv::imshow("dllTest", dllmat);
				cv::waitKey(1);

				Eigen::Matrix4f matPose = Map<Matrix4f>(pose, 4, 4);

				std::cout << "\n ------- pose: " << i << " -------- \n" << matPose
					<< "\n------------------------ " << std::endl;
			}

		}

	};
}