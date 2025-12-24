#include "../common.h"
#include <cuda_runtime.h>
#include <iostream>
#include <random>

// SGEMV: Single-precision General Matrix-Vector multiplication
// matrix_src: M * K; vector_src: K * 1; vector_dst: M * 1

__global__ void kSgemv_try(const float* __restrict__ matrix_src, const float* __restrict__ vector_src, 
    float* __restrict__ vector_dst, int M, int K) 
{
    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    if (bx >= M) return;

    float accum = 0.0f;
    // 每个线程做部分和
    for (int i = tx; i < K; i += warpSize) {
        accum += matrix_src[bx * K + i] * vector_src[i];
    }

    // warp 内归约
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        accum += __shfl_down_sync(0xFFFFFFFF, accum, offset);
    }

    if (threadIdx.x == 0) vector_dst[bx] = accum;
}
void iSgemv_try(const float* __restrict__ matrix_src, const float* __restrict__ vector_src, float* __restrict__ vector_dst, int M, int K) {
    int blockSize(32); // block的大小设置为warpSize
    int gridSize(M); // 每个block负责一行.每个 Block 只有一个线程束(Warp).在处理列数K较小的矩阵时非常高效。

    kSgemv_try<<<gridSize, blockSize>>>(matrix_src, vector_src, vector_dst, M, K);
}



__global__ void kGemv_ref(const float* __restrict__ matrix_src, const float* __restrict__ vector_src, 
    float* vector_dst, int M, int K) 
{
    const int laneId = threadIdx.x % warpSize;
    const int row = blockIdx.x;  // 0~M-1
    if (row >= M) return;

    float accum = 0.0f;
    int kIteration = CEIL(K, warpSize);  // 每个线程需要负责计算的数据个数

    for (int i = 0; i < kIteration; i++){
        int col = i * warpSize + laneId;
        accum += (col < K) ? matrix_src[row * K + col] * vector_src[col] : 0.0f;
    }

    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        accum += __shfl_down_sync(0xFFFFFFFF, accum, offset);
    }

    if(laneId == 0) vector_dst[row] = accum;
}
void iSgemv_ref(const float* __restrict__ matrix_src, const float* __restrict__ vector_src, 
    float* __restrict__ vector_dst, int M, int K) {
    int blockSize(32); // block 大小设置为warpSize
    int gridSize(M); // 每个block负责一行

    kGemv_ref<<<gridSize, blockSize>>>(matrix_src, vector_src, vector_dst, M, K);
}



void verifyResult(const float* host, const float* kernel, size_t size, double eps = 1e-3)
{
    for (size_t i = 0; i < size; ++i)
    {
        if ((host[i] - kernel[i]) > eps) {
            std::cout << "mismatch! " << "host[" << i << "]=" << host[i] 
            << " However kernel[" << i << "]=" << kernel[i] << std::endl;
            return;
        }
    }
    std::cout << "verify success!" << std::endl;

}

void printResult(const float *result, int size)
{
    for (size_t i = 0; i < size; i++) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;
}

void init_random(float* h_matrix_src, float* h_vector_src, int M, int K, float low = -5.f, float high = 5.f) {
    std::mt19937 gen(123);  // 固定 seed，方便复现
    std::uniform_real_distribution<float> dist(low, high);

    for (int i = 0; i < M; i++) {
        if (i < K) h_vector_src[i] = dist(gen);
        for (int j = 0; j < K; j++) {
            h_matrix_src[i * K + j] = dist(gen);
        }
    }
}
int main() {
    int M = 1024 * 4; 
    int K = 64 * 2;
    int repeat_times = 10;
    float total_time;

    float *h_matrix_src, *h_vector_src, *h_vector_dst, *h_vector_ref;
    h_matrix_src   = (float *)malloc(M * K * sizeof(float));
    h_vector_src   = (float *)malloc(K * sizeof(float));
    h_vector_dst   = (float *)malloc(M * sizeof(float)); 
    h_vector_ref   = (float *)malloc(M * sizeof(float)); 

    // init 
    init_random(h_matrix_src, h_vector_src, M, K);

    float *d_matrix_src, *d_vector_src, *d_vector_dst;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_matrix_src), M * K * sizeof(float)));
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_vector_src), K * sizeof(float)));
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_vector_dst), M * sizeof(float)));

    CHECK(cudaMemcpy(d_matrix_src, h_matrix_src, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_vector_src, h_vector_src, K * sizeof(float), cudaMemcpyHostToDevice));
 
    
    CHECK(cudaMemset(d_vector_dst, 0, M * sizeof(float)));
    total_time = TIME_RECORD(repeat_times, ([&]{iSgemv_ref(d_matrix_src, d_vector_src, d_vector_dst, M, K);}));
    std::cout << GREEN << std::endl  << __FILE__ << ":" << __LINE__ << 
    " [device ref]: elapsed = " << total_time / repeat_times << " ms " << RESET << std::endl;
    memset(h_vector_dst, 0, M * sizeof(float));
    CHECK(cudaMemcpy(h_vector_ref, d_vector_dst, M * sizeof(float), cudaMemcpyDeviceToHost));
    // printResult(h_vector_ref, M);

    CHECK(cudaMemset(d_vector_dst, 0, M * sizeof(float)));
    total_time = TIME_RECORD(repeat_times, ([&]{iSgemv_try(d_matrix_src, d_vector_src, d_vector_dst, M, K);}));
    std::cout << GREEN << std::endl  << __FILE__ << ":" << __LINE__ << 
    " [device try]: elapsed = " << total_time / repeat_times << " ms " << RESET << std::endl;
    memset(h_vector_dst, 0, M * sizeof(float));
    CHECK(cudaMemcpy(h_vector_dst, d_vector_dst, M * sizeof(float), cudaMemcpyDeviceToHost));
    // printResult(h_vector_dst, M);
    
    verifyResult(h_vector_ref, h_vector_dst, M);

    // cleanup
    cudaFree(d_matrix_src); cudaFree(d_vector_src); cudaFree(d_vector_dst);
    free(h_matrix_src); free(h_vector_src); free(h_vector_dst); free(h_vector_ref);

    return 0;
}
