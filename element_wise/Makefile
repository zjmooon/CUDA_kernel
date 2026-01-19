CU_APPS = element_wise 

OPENCV_INC = -I/usr/local/include/opencv4
OPENCV_LIBS = -L/usr/local/lib \
	-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

CUDA_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | sed 's/\.//') # -arch=sm_$(CUDA_ARCH)

all: ${CU_APPS}

%: %.cu
# 	nvcc -lineinfo -arch=sm_89 -o $@ $< -lcudadevrt $(OPENCV_INC) $(OPENCV_LIBS)
	nvcc -arch=sm_$(CUDA_ARCH) -G -g -O0 -o $@ $< -lcudnn -lcuda -lcudadevrt $(OPENCV_INC) $(OPENCV_LIBS) -diag-suppress 611

clean:
	rm -f ${CU_APPS} 