CU_APPS = bilateralFilter

OPENCV_INC = -I/usr/local/include/opencv4
OPENCV_LIBS = -L/usr/local/lib \
	-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

NPP_LIB_DIR = /usr/local/cuda-12.8/targets/x86_64-linux/lib
NPP_LIBS = -L$(NPP_LIB_DIR) \
	-lnppif -lnppial -lnppicc -lnppc -lnppig -lnpps -lnppim -lnppist
	
GENCODE_FLAGS := -gencode arch=compute_80,code=sm_80 \
                 -gencode arch=compute_86,code=sm_86 \
                 -gencode arch=compute_89,code=sm_89 \
                 -gencode arch=compute_89,code=compute_89 # $(GENCODE_FLAGS)
CUDA_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | sed 's/\.//') # -arch=sm_$(CUDA_ARCH)

all: ${CU_APPS}

%: %.cu

# 	nvcc -arch=sm_$(CUDA_ARCH) -G -g -O0 -o $@ $< -lcudnn -lcuda -lcudadevrt $(OPENCV_INC) $(OPENCV_LIBS) $(NPP_LIBS) -diag-suppress 611
	nvcc -arch=sm_$(CUDA_ARCH) -O3 -Xptxas=-v -lineinfo -o $@ $< -lcudnn -lcuda -lcudadevrt $(OPENCV_INC) $(OPENCV_LIBS) $(NPP_LIBS) -diag-suppress 611
# 	nvcc $(GENCODE_FLAGS) -lineinfo -o $@ $< -lcudadevrt $(OPENCV_INC) $(OPENCV_LIBS) -diag-suppress 611
# -lineinfo

clean:
	rm -f ${CU_APPS} 
