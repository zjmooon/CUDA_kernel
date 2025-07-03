CU_APPS= histogram

all: ${CU_APPS}

GENCODE_FLAGS := \
    -gencode arch=compute_89,code=sm_89 

# NVCC_FLAGS := -O2 -std=c++14 $(GENCODE_FLAGS)
NVCC_FLAGS := -std=c++14 $(GENCODE_FLAGS)

OPENCV_INCLUDE := -I/usr/local/include/opencv4
OPENCV_LIBS := -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc

%: %.cu
	nvcc $(NVCC_FLAGS) $(OPENCV_INCLUDE) $(OPENCV_LIBS) -o $@ $<
%: %.cpp
	g++ -O2 -std=c++14 $(OPENCV_INCLUDE) $(OPENCV_LIBS) -o $@ $<
clean:
	rm -f ${CU_APPS}
