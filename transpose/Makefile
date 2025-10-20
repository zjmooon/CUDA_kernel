CU_APPS= transpose

all: ${CU_APPS}

%: %.cu
	nvcc -O2 -arch=sm_89 -o $@ $< -lcudadevrt --relocatable-device-code true

clean:
	rm -f ${CU_APPS} 