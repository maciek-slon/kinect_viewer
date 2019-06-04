INCS=-Ilibfreenect/inst/include/libfreenect/ -I/opt/ros/kinetic/include/opencv-3.3.1-dev/
FLAGS=-std=c++11
LIBS=-lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lfreenect_sync -lfreenect -lopencv_highgui
LIBDIRS=-Llibfreenect/inst/lib -L/opt/ros/kinetic/lib/x86_64-linux-gnu/

all:
	g++ main.cpp -o main $(FLAGS) $(INCS) $(LIBDIRS) $(LIBS)
