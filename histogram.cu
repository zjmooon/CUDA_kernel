#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <random>
#include <iostream>
#include <opencv2/opencv.hpp>

/*
灰度直方图统计，统计图像中每个像素灰度值或颜色值出现的频率，
从而形成一张直方图（Histogram）。这可以用于图像增强（如直方图均衡化）、图像分割、特征提取等任务。
输入： unsigned char* 3840*2160*1（WHC, 16bits 0~65536）
输出： int* hist[BINNUM] 
*/
const int WIDTH = 3840;
const int HEIGHT = 2160;
const int INPUTNUM = WIDTH * HEIGHT;
const int BINNUM = 256;
const int THREADX = 32;
const int THREADY = 8;

void simulateGray(unsigned char* data, int num) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned char> distribution(0, BINNUM-1);

    for (int i = 0; i < num; ++i) {
        data[i] = distribution(gen);
    }
}

void cpuHistogram(const unsigned char* src, unsigned int* dst, const int num) {
    for (int i = 0; i < num; ++i) {
        dst[src[i]]++;
    }

    // for (int i=0; i<BINNUM; ++i) {
    //     std::cout << "gray" << i << ":" << dst[i] << " ";
    // }
    // std::cout << std::endl;
}

__global__ void gpuHistogramSMem(const unsigned char* src, unsigned int* dst, int srcWidth, int srcHeight) {
    __shared__ int shareMem[BINNUM];
    unsigned int bid = threadIdx.x + threadIdx.y * blockDim.x;
    shareMem[bid] = 0;  // 32*16 threads per block==BINNUM==256
    __syncthreads();

    unsigned int gx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int gy = threadIdx.y + blockDim.y * blockIdx.y;
    if (gx >= srcWidth || gy >= srcHeight) return;
    // 因为src是一维的，所以需要将二维网格+二维块的线程布局转换成线性索引
    unsigned int gid = gx + gy * srcWidth; // or gid = gx + gy * gridDim.x * blockDim.x;

    atomicAdd(&shareMem[src[gid]], 1);
    __syncthreads();

    atomicAdd(&dst[bid], shareMem[bid]);
}

void deviceinfo() {
    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
        deviceProp.major, deviceProp.minor);
    printf("  Total amount of global memory:                 %.2f MBytes (%llu "
           "bytes)\n", (float)deviceProp.totalGlobalMem / pow(1024.0, 3),
           (unsigned long long)deviceProp.totalGlobalMem);
    printf("  Total amount of shared memory per block:       %lu bytes\n",
           deviceProp.sharedMemPerBlock);
    printf("  Total number of registers available per block: %d\n",
           deviceProp.regsPerBlock);
    printf("  Warp size:                                     %d\n",
           deviceProp.warpSize);
    printf("  Maximum number of threads per multiprocessor:  %d\n",
           deviceProp.maxThreadsPerMultiProcessor);
    printf("  Maximum number of threads per block:           %d\n",
           deviceProp.maxThreadsPerBlock);
}

int main(int argc, char** argv)
{
    // print the device informance
    deviceinfo();

    // declare and alloc CPU memory (simulate a 4K image input)
    bool isRandom = false; 
    bool isLocalImg = true;
    unsigned char* radomArray = new unsigned char[INPUTNUM]; 
    cv::Mat localImg;

    if (isRandom) {
        simulateGray(radomArray, INPUTNUM);
    }
    
    if (isLocalImg) {
        std::string image_path = "../moon_3840_2160.jpg";
        localImg = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    }

    unsigned int* h_histo = new unsigned int[BINNUM];
    for (int i=0; i<BINNUM; i++) {
        h_histo[i] = 0;
    } 

    // declare and alloc GPU memory
    unsigned char* d_input;
    unsigned int* d_histo;
    CHECK(cudaMalloc((void **)&d_input, INPUTNUM * sizeof(unsigned char)));
    CHECK(cudaMalloc((void **)&d_histo, BINNUM * sizeof(unsigned int)));

    // transfer the CPU to the GPU
    cudaMemcpy(d_input, localImg.data, INPUTNUM * sizeof(unsigned char), cudaMemcpyHostToDevice); 
    cudaMemcpy(d_histo, 0,             BINNUM * sizeof(unsigned int),     cudaMemcpyHostToDevice);   
    
    dim3 block(THREADX, THREADY);
    dim3 grid((WIDTH + THREADX - 1) / THREADX,
                  (HEIGHT + THREADY - 1) / THREADY);

    // launch the kernel
    int kernelFun = 0;
    if (argc == 2) {
        kernelFun = atoi(argv[1]);
    }

    double start, elaps;
    cudaEvent_t k_start, k_stop;
    bool isWarmUp = true;
    float elapsed_ms = 0.0f;
    cudaEventCreate(&k_start);
    cudaEventCreate(&k_stop);

    switch (kernelFun)
    {
    case 0:
        start = seconds();
        cpuHistogram(localImg.data, h_histo, INPUTNUM);
        elaps = seconds() - start;
        std::cout << "Host elapsed " << elaps * 1000 << "ms" << std::endl;
        break;
    case 1:
        if (isWarmUp) {
            gpuHistogramSMem<<<grid, block>>>(d_input, d_histo, WIDTH, HEIGHT);
            cudaMemcpy(d_histo, 0,             BINNUM * sizeof(unsigned int),     cudaMemcpyHostToDevice);
        }
        cudaEventRecord(k_start, 0);
        gpuHistogramSMem<<<grid, block>>>(d_input, d_histo, WIDTH, HEIGHT);
        cudaEventRecord(k_stop, 0);
        cudaEventSynchronize(k_stop);
        
        cudaEventElapsedTime(&elapsed_ms, k_start, k_stop);
        std::cout << "Kernel execution elapsed: " << elapsed_ms << " ms" << std::endl;

        cudaMemcpy(h_histo, d_histo, BINNUM * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        for (int i=0; i<BINNUM; ++i) {
            std::cout << "gray" << i << ":" << h_histo[i] << " ";
        }
        std::cout << std::endl; 
        break;        
    case 2:
        
        break;
    default:
        std::cout << "do Nothing" << std::endl;
        break;
    }

    delete[] radomArray;
    delete[] h_histo;
    CHECK(cudaFree(d_histo));
    CHECK(cudaFree(d_input));
    cudaEventDestroy(k_start);
    cudaEventDestroy(k_stop);

    return 0;
} 