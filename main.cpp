#include <libfreenect_sync.h>
#include <opencv2/opencv.hpp>

#include <chrono>
#include <vector>
#include <string>

#include "date.h"

std::string strip(const std::string & str) {
	std::string ret = str;
	ret.erase( std::remove(ret.begin(), ret.end(), '\r'), ret.end() );
	return ret;
}


template<typename Out>
void split(const std::string &s, char delim, Out result) {
	std::stringstream ss;
	ss.str(s);
	std::string item;
	while (std::getline(ss, item, delim)) {
		*(result++) = strip(item);
	}
}

std::vector<std::string> split(const std::string &s, char delim) {
	std::vector<std::string> elems;
	split(s, delim, std::back_inserter(elems));
	return elems;
}



void putTextCentered(cv::Mat img, const std::string & str, cv::Point anchor, int fontFace, double fontScale, cv::Scalar color, int thickness = 1, int type = cv::LINE_AA) {
	int baseline;
	cv::Size fs = cv::getTextSize(str, fontFace, fontScale, thickness, &baseline);
	int x = anchor.x - fs.width / 2;
	int y = anchor.y + fs.height / 2;
	cv::putText(img, str, cv::Point(x, y), fontFace, fontScale, color, thickness, type);
}

void putTexts(cv::Mat img, const std::string & str, cv::Point origin, int fontFace, double fontScale, cv::Scalar color, float inter = 1.0, int thickness = 1, int type = cv::LINE_AA) {
	auto lines = split(str, '\n');
	cv::Point ori = origin;
	for (int i = 0; i < lines.size(); ++i) {
		cv::putText(img, lines[i], ori , fontFace, fontScale, color, thickness, type);
		ori.y += 20 * fontScale * inter;
	}
}


std::string timestamp() {
	using namespace date;
	using namespace std::chrono;
	auto now = date::floor<milliseconds>(system_clock::now());
	return date::format("%Y-%m-%d_%H-%M-%S", now);
}

cv::Mat getHist(cv::Mat depth, float rng = 6000) {
	float range[] = { 1, rng } ;
	const float* histRange = { range };
	int histSize = 1200;
	bool uniform = true; 
	bool accumulate = false;

	cv::Mat hist;

	cv::calcHist( &depth, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );

	return hist;
}

cv::Mat drawHist(cv::Mat hist) {
	int hh = 100, hw = 1200;
	cv::Mat hist_image = cv::Mat::zeros(hh+10, hw, CV_8UC3);
	cv::normalize(hist, hist, 0, hh, cv::NORM_MINMAX, -1, cv::Mat());
	for (int i = 0; i < hist.size().height; i++) {
		cv::line(hist_image, cv::Point(i, hh+10), cv::Point( i, hh - cvRound(hist.at<float>(i))), cv::Scalar::all(255));
	}
	return hist_image;
}

void putOn(cv::Mat dst, cv::Mat src, cv::Point origin) {
	src.copyTo(dst(cv::Rect(origin.x,origin.y,src.cols, src.rows)));
}

void drawCanvas(cv::Mat canvas) {
	int hist_x = 30, hist_y = 100;
	int hh = 100 + 10;
	for (int i = 0; i <= 24; ++i) {
		cv::line(canvas, {hist_x + i*50, hist_y + hh}, {hist_x + i*50, hist_y + hh+3}, cv::Scalar::all(255));
		putTextCentered(canvas, std::to_string(i*250), {hist_x + i*50, hist_y + hh+10}, cv::FONT_HERSHEY_PLAIN, 0.8, cv::Scalar::all(255));
	}
	putTextCentered(canvas, "mm", {hist_x + 1200 + 30, hist_y + hh+10}, cv::FONT_HERSHEY_PLAIN, 0.8, cv::Scalar::all(255));

	std::string help = 
		"S - save images\n"
		"A - auto range\n"
		"Q - quit"
	;

	putTexts(canvas, help, {10, 20}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar::all(255), 2);
	
	cv::rectangle(canvas, cv::Rect(5, 4, 200, 82), cv::Scalar::all(255), 1);
	
	putTexts(canvas, "R:\nG:\nB:\nD:", {1080, 20}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar::all(255), 2);
	cv::rectangle(canvas, cv::Rect(1075, 4, 200, 82), cv::Scalar::all(255), 1);
}

struct mouse_pos {
	mouse_pos() : x(-1), y(-1) {}
	int x;
	int y;
};

static void onMouse( int event, int x, int y, int, void* data) {
	mouse_pos * mp = (mouse_pos*)data;
	y = y - 240;
	if (x > 640) x = x - 640;
	mp->x = x;
	mp->y = y;
}

int main(int argc, char * argv[]) {
	short *depth = 0;
	char *rgb = 0;
	uint32_t ts;
	int ret;

	int depth_min = 500;
	int depth_range = 1500;

	int blend_ratio = 50;

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


	cv::namedWindow("KinectViewer");
	cv::createTrackbar("min", "KinectViewer", &depth_min, 10000, NULL);
	cv::createTrackbar("range", "KinectViewer", &depth_range, 10000, NULL);
	cv::createTrackbar("blend", "KinectViewer", &blend_ratio, 100, NULL);
	
	mouse_pos mp;
    cv::setMouseCallback( "KinectViewer", onMouse, &mp );


	cv::Mat canvas(720, 1280, CV_8UC3, cv::Scalar::all(0));
	drawCanvas(canvas);

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

		cv::Mat out_rgb, tmp_rgb;
		cv_rgb.copyTo(tmp_rgb, depth_mask);
		cv::addWeighted(cv_rgb, 0.01 * blend_ratio, tmp_rgb, 0.01 * (100-blend_ratio), 0, out_rgb);


		cv::Mat hist = getHist(cv_depth);
		cv::Mat hist_img = drawHist(hist);
		
		cv::Mat hist_overlay = cv::Mat::zeros(hist_img.size(), CV_8UC3);
		cv::Mat hist_depth = cv::Mat::zeros(hist_img.size(), CV_16UC1);
		for (int i = 0; i < 1200; ++i) {
			cv::line(hist_depth, {i, 0}, {i, hist_depth.size().height}, cv::Scalar::all(i*5));
		}
		hist_depth = hist_depth - depth_min;
		cv::convertScaleAbs(hist_depth, hist_depth, 255./depth_range);
		cv::applyColorMap(hist_depth, hist_overlay, cv::COLORMAP_JET);
		
		hist_overlay.copyTo(hist_img, hist_img);
		cv::line(hist_img, {depth_min/5, 100}, {depth_min/5, 110}, cv::Scalar::all(255));
		cv::line(hist_img, {(depth_min+depth_range)/5, 100}, {(depth_min+depth_range)/5, 110}, cv::Scalar::all(255));

		cv::rectangle(canvas, cv::Rect(1100, 5, 150, 80), cv::Scalar::all(0), -1);
		if (mp.y >= 0) {
			int d = cv_depth.at<short>(mp.y, mp.x);
			auto bgr = cv_rgb.at<cv::Vec3b>(mp.y, mp.x);
			std::string pixel_str = std::to_string(bgr[2]) + "\n" + std::to_string(bgr[1]) + "\n" +
				std::to_string(bgr[0]) + "\n" + std::to_string(d);
			putTexts(canvas, pixel_str, {1100, 20}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar::all(255), 2);
			cv::rectangle(canvas, {1150,5,60,60}, cv::Scalar(bgr), -1);
			cv::rectangle(canvas, {1150,65,60,20}, col_depth.at<cv::Vec3b>(mp.y, mp.x), -1);
		}

		putOn(canvas, out_rgb, {0, 240});
		putOn(canvas, col_depth, {640, 240});
		putOn(canvas, hist_img, {30, 100});
		cv::imshow("KinectViewer", canvas);

		char ch = cv::waitKey(15);

		switch(ch) {
			case 27:
			case 'q':
			case 'Q':
				return 0;
			case 's':
			case 'S': {
					auto ts = timestamp();
					cv::imwrite(ts + "_d.png", cv_depth);
					cv::imwrite(ts + "_c.png", cv_rgb);
					break;
				}
			case 'a': {
					float hist_sum = cv::sum(hist)[0];
					cv::Mat accumulatedHist = hist.clone();
					for (int i = 1; i < hist.size().height; i++) {
						accumulatedHist.at<float>(i) += accumulatedHist.at<float>(i - 1);
						if (accumulatedHist.at<float>(i) < 0.001 * hist_sum) depth_min = i*5;
						if (accumulatedHist.at<float>(i) < 0.999 * hist_sum) depth_range = i*5 - depth_min;
					}
					cv::setTrackbarPos("min", "KinectViewer", depth_min);
					cv::setTrackbarPos("range", "KinectViewer", depth_range);
					break;
				}
		}	
	}
	
	return 0;
}
