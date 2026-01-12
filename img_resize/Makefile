CU_APPS = rgb_uint8_resize

OPENCV_INC = -I/usr/local/include/opencv4
OPENCV_LIBS = -L/usr/local/lib \
	-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs
	
GENCODE_FLAGS := -gencode arch=compute_80,code=sm_80 \
                 -gencode arch=compute_86,code=sm_86 \
                 -gencode arch=compute_89,code=sm_89 \
                 -gencode arch=compute_89,code=compute_89 # $(GENCODE_FLAGS)
CUDA_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | sed 's/\.//') # -arch=sm_$(CUDA_ARCH)

all: ${CU_APPS}

%: %.cu

# 	nvcc -arch=sm_$(CUDA_ARCH) -G -g -O0 -o $@ $< -lcudnn -lcuda -lcudadevrt $(OPENCV_INC) $(OPENCV_LIBS) -diag-suppress 611
	nvcc -arch=sm_$(CUDA_ARCH) -Xptxas=-v -o $@ $< -lcudnn -lcuda -lcudadevrt $(OPENCV_INC) $(OPENCV_LIBS) -diag-suppress 611
# 	nvcc $(GENCODE_FLAGS) -o $@ $< -lcudadevrt $(OPENCV_INC) $(OPENCV_LIBS) -diag-suppress 611
# -lineinfo

clean:
	rm -f ${CU_APPS} 
