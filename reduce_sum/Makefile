CU_APPS= reduceInteger 

all: ${CU_APPS}

# GENCODE_FLAGS := \
#     -gencode arch=compute_89,code=sm_89 

# NVCC_FLAGS := -O2 -std=c++14 $(GENCODE_FLAGS)

# %: %.cu
# 	nvcc $(NVCC_FLAGS) -o $@ $<
# %: %.c
# 	g++ -O2 -std=c++14 -o $@ $<
# clean:
# 	rm -f ${CU_APPS}

%: %.cu
	nvcc -O2 -arch=sm_89 -o $@ $< -lcudadevrt --relocatable-device-code true
%: %.c
	g++ -O2 -std=c++14 -o $@ $<
clean:
	rm -f ${CU_APPS} 