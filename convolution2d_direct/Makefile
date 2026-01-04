CU_APPS = convolution2d_direct 

OPENCV_INC = -I/usr/local/include/opencv4
OPENCV_LIBS = -L/usr/local/lib \
	-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

all: ${CU_APPS}

%: %.cu
	nvcc -arch=sm_89 -G -g -O0 -o $@ $< -lcudadevrt $(OPENCV_INC) $(OPENCV_LIBS) -diag-suppress 611
# 	nvcc -arch=sm_89 -lineinfo -o $@ $< -lcudadevrt $(OPENCV_INC) $(OPENCV_LIBS) -diag-suppress 611

clean:
	rm -f ${CU_APPS} 