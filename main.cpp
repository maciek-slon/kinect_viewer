#include <libfreenect_sync.h>
#include <opencv2/opencv.hpp>

#include <chrono>
#include <string>

#include "date.h"

std::string timestamp() {
	using namespace date;
	using namespace std::chrono;
	auto now = date::floor<milliseconds>(system_clock::now());
	return date::format("%Y-%m-%d_%H-%M-%S", now);
}

int main(int argc, char * argv[]) {
	short *depth = 0;
	char *rgb = 0;
	uint32_t ts;
	int ret;

	int depth_min = 0;
	int depth_range = 5000;

	int index = 0;

	bool sim = false;

	const cv::String keys =
		"{help h usage ? |      | print this message   }"
		"{@rgb           |      | rgb image            }"
		"{@depth         |      | depth image          }"
		"{device         |0     | device id            }"
        ;

	cv::CommandLineParser parser(argc, argv, keys);
	parser.about("KinectViewer v0.0.1");
	if (parser.has("help"))
	{
	    parser.printMessage();
	    return 0;
	}
	index = parser.get<int>("device");
	cv::String img1 = parser.get<cv::String>(0);
	cv::String img2 = parser.get<cv::String>(1);
	if (!parser.check())
	{
	    parser.printErrors();
	    return 0;
	}

	cv::Mat sim_rgb, sim_depth;
	if (!img1.empty() && !img2.empty()) {
		std::cout << "Reading images" << std::endl;
		sim_rgb = cv::imread(img1);
		if (sim_rgb.empty()) {
			std::cerr << "Can't read rgb image: " << img1 << std::endl;
			return -1;
		}

		sim_depth = cv::imread(img2, -1);
		if (sim_rgb.empty()) {
			std::cerr << "Can't read depth image: " << img1 << std::endl;
			return -1;
		}
		sim = true;
	}


	cv::namedWindow("depth");
	cv::createTrackbar("min", "depth", &depth_min, 10000, NULL);
	cv::createTrackbar("range", "depth", &depth_range, 10000, NULL);


	while(1) {
		cv::Mat cv_rgb, cv_depth;
		if (sim) {
			cv_rgb = sim_rgb.clone();
			cv_depth = sim_depth.clone();
		} else {
			ret = freenect_sync_get_video((void**)&rgb, &ts, index, FREENECT_VIDEO_RGB);
			cv::Mat tmp_rgb(480, 640, CV_8UC3, rgb);
			cv::cvtColor(tmp_rgb, cv_rgb, cv::COLOR_RGB2BGR);

	
			ret = freenect_sync_get_depth((void**)&depth, &ts, index, FREENECT_DEPTH_REGISTERED);
			cv::Mat tmp_depth(480, 640, CV_16UC1, depth);
			cv_depth = tmp_depth;
		}
		
		cv::Mat out_depth = cv_depth - depth_min; 
		cv::Mat valid_mask = (cv_depth != 0);
		cv::Mat depth_mask = (cv_depth >= depth_min) & (cv_depth <= depth_min + depth_range) & valid_mask;
		cv::convertScaleAbs(out_depth, out_depth, 255./depth_range);
		cv::Mat col_depth, tmp_depth;
		cv::applyColorMap(out_depth, tmp_depth, cv::COLORMAP_JET);
		tmp_depth.copyTo(col_depth, valid_mask);

		cv::imshow("rgb", cv_rgb);
		cv::imshow("depth", col_depth);

		char ch = cv::waitKey(15);

		switch(ch) {
			case 27:
				return 0;
			case 's':
				using namespace std::chrono;
				auto timestamp = duration_cast< milliseconds >(system_clock::now().time_since_epoch()).count();
				cv::imwrite(std::to_string(timestamp) + "_d.png", cv_depth);
				cv::imwrite(std::to_string(timestamp) + "_c.png", cv_rgb);		
		}	
	}
	
	return 0;
}
