#include "../common.h"
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
const int BLOCK_X = 32;
const int BLOCK_Y = 8;

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

__global__ void warmingup(const unsigned char *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned char ia, ib;
    ia = ib = 0;

    if ((tid / warpSize) % 2 == 0)
    {
        ia = 1;
    }
    else
    {
        ib = 2;
    }
}

/*
*参考CPU的方式，区别在于需要使用原子加法做线程同步
*/
__global__ void gpuHistogramNaive(const unsigned char* src, unsigned int* dst, int srcWidth, int srcHeight) {
    unsigned int gx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int gy = threadIdx.y + blockIdx.y * blockDim.y;

    unsigned int gid = gy * blockDim.x * gridDim.x + gx;

    atomicAdd(&dst[src[gid]], 1); 
} 


/*
*提高线程利用率，一个线程处理多个数据，提高访存吞吐量(32*1-->32*4bytes == L1 memory transaction 128bytes)
*向量化处理，将uchar* 转换成 uchar4*，在CPU端进行转换
*/ 
__global__ void gpuHistogramVectorization(const uchar4* src, unsigned int* dst, int srcWidth, int srcHeight) {
    unsigned int gx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int gy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int gid = gx + gy * blockDim.x * gridDim.x;

    if (gid >= srcWidth * srcHeight) return;

    atomicAdd(&dst[src[gid].x], 1);
    atomicAdd(&dst[src[gid].y], 1);
    atomicAdd(&dst[src[gid].z], 1);
    atomicAdd(&dst[src[gid].w], 1);
}


/*
*共享内存优化 思路是将图像按照block的大小切块，分别在block里面统计小图片的直方图，然后再将所有的直方图相加
*巧妙或者兼容性差的点在于共享内存的数组大小BINNUM和一个block内总线程数是一样的，都是256。这样刚好可以满足一个线程给一个共享内存的值初始化。
*如果BINNUM > blockSize, 每个线程需要给多个共享内存值初始化。比如在本内核函数中，需要统计的灰度值为0~65535，一般以块大小blockDim.x*blockDim.y为stride
    unsigned int bid = threadIdx.y * blockDim.x + threadIdx.x;
    for (int i = bid; i < BINNUM; i += blockDim.x * blockDim.y) {
        shareMem[i] = 0;
    }
*如果BINNUM < blockSize, 那共享内存的初始化并不需要所有的线程都工作。比如在本内核函数中，
    申请的blockSize大于32*8就可以通过一些if语句来判断当前块内线程索引是否超过了BINNUM.
*/ 
__global__ void gpuHistogramSMem(const unsigned char* src, unsigned int* dst, int srcWidth, int srcHeight) {
    __shared__ int shareMem[BINNUM];
    unsigned int bid = threadIdx.x + threadIdx.y * blockDim.x;
    shareMem[bid] = 0;  // 32*8 threads per block==BINNUM==256
    __syncthreads();

    unsigned int gx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int gy = threadIdx.y + blockDim.y * blockIdx.y;
    if (gx >= srcWidth || gy >= srcHeight) return;
    // 因为src是一维的，所以需要将二维网格+二维块的线程布局转换成线性索引
    unsigned int gid = gx + gy * srcWidth; // or gid = gx + gy * gridDim.x * blockDim.x;

    // 比较难理解的点在于各种索引的转换，线程全局索引->图像灰度值->block统计
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
    CHECK(cudaMemcpy(d_input, localImg.data, INPUTNUM * sizeof(unsigned char), cudaMemcpyHostToDevice)); 
    CHECK(cudaMemcpy(d_histo, h_histo,             BINNUM * sizeof(unsigned int),    cudaMemcpyHostToDevice));   
    
    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid(CEIL(WIDTH, BLOCK_X),
                  CEIL(HEIGHT, BLOCK_Y));
               
    uchar4* h_uchar4_input = reinterpret_cast<uchar4*>(localImg.data);
    uchar4* d_uchar4_input;

    CHECK(cudaMalloc((void **)&d_uchar4_input, (INPUTNUM >> 2) * sizeof(uchar4)));
    CHECK(cudaMemcpy(d_uchar4_input, h_uchar4_input, (INPUTNUM >> 2) * sizeof(uchar4), cudaMemcpyHostToDevice));

    // launch the kernel
    int kernelFun = 0;
    if (argc == 2) {
        kernelFun = atoi(argv[1]);
    }

    double start, elaps;
    cudaEvent_t k_start, k_stop;
    float elapsed_ms = 0.0f;
    CHECK(cudaEventCreate(&k_start));
    CHECK(cudaEventCreate(&k_stop));

    // run a warmup kernel to remove overhead
    CHECK(cudaDeviceSynchronize());
    warmingup<<<grid, block>>>(d_input);
    CHECK(cudaDeviceSynchronize());


    switch (kernelFun)
    {
    case 0:
        start = seconds();
        cpuHistogram(localImg.data, h_histo, INPUTNUM);
        elaps = seconds() - start;
        std::cout << "Host elapsed " << elaps * 1000 << "ms" << std::endl;
        break;
    case 1:
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(k_start, 0));
        gpuHistogramNaive<<<grid, block>>>(d_input, d_histo, WIDTH, HEIGHT);
        CHECK(cudaEventRecord(k_stop, 0));
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventSynchronize(k_stop));
        
        CHECK(cudaEventElapsedTime(&elapsed_ms, k_start, k_stop));
        std::cout << "gpuHistogramNaive execution elapsed: " << elapsed_ms << " ms" << std::endl;

        CHECK(cudaMemcpy(h_histo, d_histo, BINNUM * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        for (int i=0; i<BINNUM; ++i) {
            std::cout << "gray" << i << ":" << h_histo[i] << " ";
        }
        std::cout << std::endl; 
        break;
    case 2:
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(k_start, 0));
        gpuHistogramSMem<<<grid, block>>>(d_input, d_histo, WIDTH, HEIGHT);
        CHECK(cudaEventRecord(k_stop, 0));
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventSynchronize(k_stop));
        
        CHECK(cudaEventElapsedTime(&elapsed_ms, k_start, k_stop));
        std::cout << "gpuHistogramSMem execution elapsed: " << elapsed_ms << " ms" << std::endl;

        CHECK(cudaMemcpy(h_histo, d_histo, BINNUM * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        for (int i=0; i<BINNUM; ++i) {
            std::cout << "gray" << i << ":" << h_histo[i] << " ";
        }
        std::cout << std::endl; 
        break;        
    case 3:
        block = dim3(BLOCK_X, BLOCK_Y);
        grid  = dim3(CEIL(CEIL(WIDTH, 4), BLOCK_X), CEIL(HEIGHT, BLOCK_Y));

        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(k_start, 0));
        gpuHistogramVectorization<<<grid, block>>>(d_uchar4_input, d_histo, WIDTH, HEIGHT);
        CHECK(cudaEventRecord(k_stop, 0));
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventSynchronize(k_stop));

        CHECK(cudaEventElapsedTime(&elapsed_ms, k_start, k_stop));
        std::cout << "gpuHistogramVectorization execution elapsed: " << elapsed_ms << " ms" << std::endl;

        CHECK(cudaMemcpy(h_histo, d_histo, BINNUM * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        for (int i=0; i<BINNUM; ++i) {
            std::cout << "gray" << i << ":" << h_histo[i] << " ";
        }
        std::cout << std::endl; 
        break;
    default:
        std::cout << "do Nothing" << std::endl;
        break;
    }

    delete[] radomArray;
    delete[] h_histo;
    CHECK(cudaFree(d_histo));
    CHECK(cudaFree(d_input));
    CHECK(cudaEventDestroy(k_start));
    CHECK(cudaEventDestroy(k_stop));

    return 0;
} 