#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include "../common.h"

__global__ void sgemmNaive(float* A, float* B, float* C, int M, int N, int K) {
    uint gx = threadIdx.x + blockIdx.x * blockDim.x;
    uint gy = threadIdx.y + blockIdx.y * blockDim.y;

    if (gx >= N || gy >= M) return; // C:(M行×N列)

    float value = 0.0f;
    for (int i=0; i<K; ++i) {
        value += A[gy * K + i] * B[i * N + gx];
    }

    C[gy * N + gx] = value;
}


template<const int BLOCK_SIZE>
__global__ void sgemmSharedMem(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    const int BM = BLOCK_SIZE;
    const int BN = BLOCK_SIZE;
    const int BK = BLOCK_SIZE;

    int bx = blockIdx.x; // 当前处理C中哪一列tile
    int by = blockIdx.y; // 当前处理C中哪一行tile
    int tx = threadIdx.x % BN;  // 当前线程在tile中横向的坐标
    int ty = threadIdx.x / BN;  // 当前线程在tile中竖向的坐标

    // 申请共享内存空间
    // NVIDIA GeForce GTX 4080super's sharedMemPerBlock is 49152 bytes
    // 1 float takes 4 Bytes, so (BM*BK + BK*BN) should <= 49152/4 = 12288 bytes
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // 移动到当前block
    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    float tmp = 0.;
    for (int k = 0; k < K; k += BK) {  // 窗口滑动
        // 缓存A_tile和B_tile
        As[ty * BK + tx] = A[ty * K + tx];
        Bs[ty * BN + tx] = B[ty * N + tx];
        // 同步所有线程缓存完成
        __syncthreads();  // 同步同一个线程块(block)中的线程，执行到同一个点
        // 移动A,B指针到下一个矩阵块
        A += BK;
        B += BK * N;
        for (int i = 0; i < BK; i++) {
            tmp += As[ty * BK + i] * Bs[i * BN + tx];
        }
        // FMA计算需要读取缓存数据，在新一轮写入缓存前进行同步，确保所有线程计算完成
        __syncthreads();
    }
    C[ty * N + tx] = alpha * tmp + beta * C[ty * N + tx];
}

// template instantiation declaration
template __global__ void sgemmSharedMem<16>(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);
template __global__ void sgemmSharedMem<32>(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);
template __global__ void sgemmSharedMem<64>(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);

// Note: pay attention to the `sharedMemPerBlock`,
// for example, when there is a template instantiation declaration like below:
// template __global__ void sgemm_v2<128>(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);
// compiler will throw error like below:
// ptxas error   : Entry function '_Z8sgemm_v2ILi128EEviiifPfS0_fS0_' uses too much shared data (0x20000 bytes, 0xc000 max)



// const int BLOCK_SIZE = 32;
// dim3 block(BLOCK_SIZE, BLOCK_SIZE);
// dim3 grid(CEIL(N,BLOCK_SIZE), CEIL(M,BLOCK_SIZE));  // 根据C矩阵的形状(M行N列)切块
// sgemm<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

const int BLOCK_SIZE = 32;
__global__ void sgemm(float* A, float* B, float* C, int M, int N, int K) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    if (idx >= M || idy >= N) return; // ? if (idx >= N || idy >= M) return;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    const int BM = BLOCK_SIZE;
    const int BN = BLOCK_SIZE;
    const int BK = BLOCK_SIZE;
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // 初始化block tile起始位置
    A = &A[(by * BM) * K];
    B = &B[bx * BN];
    C = &C[(by * BM) * N + bx * BN];

    float accum = 0.0f;
    for (int k = 0; k < K; k += BK) {
        // 搬运 global ==> shared
        As[ty * BK + tx] = A[ty * K + tx];
        Bs[ty * BN + tx] = B[ty * N + tx];
        __syncthreads();
        A = A + BK;
        B = B + BK * N;
        for (int i = 0; i < BK; i++) {
            accum += As[ty * BK + i] * Bs[i * BN + tx];
        }
        __syncthreads();
    }

    C[ty * N + tx] = accum;
}