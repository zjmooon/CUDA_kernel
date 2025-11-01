CU_APPS= sgemm 

all: ${CU_APPS}

%: %.cu
	nvcc -lineinfo -arch=sm_89 -o $@ $< -lcudadevrt

clean:
	rm -f ${CU_APPS} 