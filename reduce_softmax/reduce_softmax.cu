#include "../common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <random>

// softmax(xi) = exp(xi - M) / exp(xi - M).sum()
void cpu_softmax(const float *input, float *output, int size) {
    float M = *(std::max_element(input, input + size));

    float sum = 0.f;
    for (int i = 0; i < size; i++) {
        output[i] = expf(input[i] - M);
        sum += output[i];
    }

    for (int i = 0; i < size; i++) {
        output[i] /= sum;
        // printf("%f ", output[i]);
    }
}


__device__ static float atomicMaxFloat(float* addr, float value) {
    int* addr_i = (int*)addr;
    int old = *addr_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_i, assumed,
            __float_as_int(fmaxf(value, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void k_Max(const float *input, float *Max, int N) 
{
    __shared__ float s_data[32];
    int g_idx = threadIdx.x + blockDim.x * blockIdx.x;
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;

    float val = (g_idx < N) ? input[g_idx] : (-FLT_MAX);
    #pragma unroll
    for (int stride = warpSize/2; stride > 0; stride >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, stride));
    }

    if (laneId == 0) s_data[warpId] = val;
    __syncthreads();

    if (warpId == 0) {
        int warpNum = blockDim.x / warpSize;
        val = (laneId < warpNum) ? s_data[laneId] : (-FLT_MAX); 
        #pragma unroll
        for (int stride = warpSize/2; stride > 0; stride >>= 1) {
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, stride));
        } 
        if (laneId == 0) atomicMaxFloat(Max, val);
    }
}
__global__ void k_Sum(const float *input, float *Sum, float *Max, int N) 
{
    __shared__ float s_data[32];
    int g_idx = threadIdx.x + blockDim.x * blockIdx.x;
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;

    float val = (g_idx < N) ? expf(input[g_idx] - (*Max)) : 0.f;
    #pragma unroll
    for (int stride = warpSize/2; stride > 0; stride >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, stride);
    }

    if (laneId == 0) s_data[warpId] = val;
    __syncthreads();

    if (warpId == 0) {
        int warpNum = blockDim.x / warpSize;
        val = (laneId < warpNum) ? s_data[laneId] : 0; 
        #pragma unroll
        for (int stride = warpSize/2; stride > 0; stride >>= 1) {
            val +=  __shfl_down_sync(0xffffffff, val, stride);
        } 
        if (laneId == 0) atomicAdd(Sum, val);
    }
}
__global__ void k_Softmax(const float *input, float *output, float *Sum, float *Max, int N) 
{
    int g_idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (g_idx < N) output[g_idx] = expf(input[g_idx] - (*Max)) / (*Sum);
}
void iSoftmax_WarpReduce(const float *input, float *output, int N) {
    dim3 blockSize = 256;
    dim3 gridSize  = CEIL(N, blockSize.x);

    float *Max = nullptr, *Sum = nullptr;
    cudaMalloc(reinterpret_cast<void **>(&Max), sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&Sum), sizeof(float));
    float neg_inf = -FLT_MAX;
    float zero = 0.f;
    cudaMemcpy(Max, &neg_inf, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Sum, &zero, sizeof(float), cudaMemcpyHostToDevice);

    
    k_Max<<<gridSize, blockSize>>>(input, Max, N);
    k_Sum<<<gridSize, blockSize>>>(input, Sum, Max, N);
    k_Softmax<<<gridSize, blockSize>>>(input, output, Sum, Max, N);
}

// Fusion version: CUDA编程只提供BLock级别的同步机制，无法在Block内部实现线程间的完全同步
// 如max和sum不能够做到完全同步，因此只能通过原子操作来保证正确性
__global__ void kSoftmax_WarpReduce(const float *input, float *output, float *Max, float *Sum, int N) 
{
    __shared__ float s_data[32];
    int g_idx = threadIdx.x + blockDim.x * blockIdx.x;
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;

    if (g_idx >= N) return;

    float val = (g_idx < N) ? input[g_idx] : (-FLT_MAX);
    #pragma unroll
    for (int stride = warpSize/2; stride > 0; stride >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, stride));
    }

    if (laneId == 0) s_data[warpId] = val;
    __syncthreads();

    if (warpId == 0) {
        int warpNum = blockDim.x / warpSize;
        val = (laneId < warpNum) ? s_data[laneId] : (-FLT_MAX); 
        #pragma unroll
        for (int stride = warpSize/2; stride > 0; stride >>= 1) {
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, stride));
        } 
        if (laneId == 0) atomicMaxFloat(Max, val);
    }
    __syncthreads(); // 确保Max已经更新完成

    val = (g_idx < N) ? expf(input[g_idx] - (*Max)) : 0.f;
    #pragma unroll
    for (int stride = warpSize/2; stride > 0; stride >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, stride);
    }

    if (laneId == 0) s_data[warpId] = val;
    __syncthreads();

    if (warpId == 0) {
        int warpNum = blockDim.x / warpSize;
        val = (laneId < warpNum) ? s_data[laneId] : 0; 
        #pragma unroll
        for (int stride = warpSize/2; stride > 0; stride >>= 1) {
            val +=  __shfl_down_sync(0xffffffff, val, stride);
        } 
        if (laneId == 0) atomicAdd(Sum, val);
    }
    __syncthreads(); // 确保Sum已经更新完成

    if (g_idx < N) output[g_idx] = expf(input[g_idx] - (*Max)) / (*Sum);
}
void iSoftmax_WarpReduce_fusion(const float *input, float *output, int N) {
    dim3 blockSize = 256;
    dim3 gridSize  = CEIL(N, blockSize.x);

    float *Max = nullptr, *Sum = nullptr;
    cudaMalloc(reinterpret_cast<void **>(&Max), sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&Sum), sizeof(float));
    float neg_inf = -FLT_MAX;
    float zero = 0.f;
    cudaMemcpy(Max, &neg_inf, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Sum, &zero, sizeof(float), cudaMemcpyHostToDevice);

    kSoftmax_WarpReduce<<<gridSize, blockSize>>>(input, output, Max, Sum, N);
}


/*
* softmax(xi) = exp(xi - M) / exp(xi - M).sum()
* 以行为单位做softmax
*/ 
__global__ void kSgemv_softmax(const float* __restrict__ input, float* __restrict__ output, int M, int N) 
{
    const int tx = threadIdx.x;
    const int bx = blockIdx.x;

    /* __shared__ float max_val_shared; // Block内共享变量
    __shared__ float sum_val_shared; */

    // max
    // 先求出每个线程负责数据的max_val
    float max_val = -FLT_MAX;
    for (int i = tx; i < N; i += warpSize) {
        max_val = fmaxf(input[bx * N + i], max_val);
    }
    // warp内归约求出每行最大值
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_xor_sync(0XFFFFFFFF, max_val, offset));
    } 
    // __shfl_xor_sync: 已经求得max_val(每个线程中)
    /* if (tx == 0) {
        max_val_shared = max_val;
    } */

    // sum
    float sum_val = 0.0f;
    for (int i = tx; i < N; i += warpSize) {
        /* sum_val += expf(input[bx * N + i] - max_val_shared); */
        sum_val += expf(input[bx * N + i] - max_val);
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum_val += __shfl_xor_sync(0XFFFFFFFF, sum_val, offset);
    }
    // __shfl_xor_sync: 已经求得sum_val(每个线程中)
    /* if (tx == 0) {
        sum_val_shared = sum_val;
    } */

    // 对一行数据的所有数据做softmax
    for (int i = tx; i < N; i += warpSize) {
        output[bx * N + i] = expf(input[bx * N + i] - max_val) / sum_val;
    }

}
void iSgemv_softmax(const float* __restrict__ input, float* __restrict__ output, int M, int N) {
    int blockSize(32); // block的大小设置为warpSize
    int gridSize(M); // 每个block负责一行.每个 Block 只有一个线程束(Warp).在处理列数K较小的矩阵时非常高效

    kSgemv_softmax<<<gridSize, blockSize>>>(input, output, M, N);
}



void verifyResult(const float* host, const float* kernel, size_t size, double eps = 1e-3)
{
    double max_abs_err = 0.0;
    double sum_abs_err = 0.0;
    size_t num_bad = 0;

    for (size_t i = 0; i < size; ++i)
    {
        double diff = std::fabs(static_cast<double>(host[i]) - static_cast<double>(kernel[i]));
        double abs_ref = std::fabs(static_cast<double>(host[i]));
        double rel_err = (abs_ref > 1e-6) ? diff / abs_ref : diff;

        if (rel_err > eps) {
            ++num_bad;
            std::cout << i << ": " << host[i] << ", kernel " << kernel[i] << std::endl;
            // return;
        }

        if (diff > max_abs_err) {
            max_abs_err = diff;
        }

        sum_abs_err += diff;
    }

    double mean_abs_err = sum_abs_err / static_cast<double>(size);

    std::cout << "Verification Result:\n"
              << std::scientific << std::setprecision(6)
              << "  Max abs error   = " << max_abs_err << "\n"
              << "  Mean abs error  = " << mean_abs_err << "\n"
              << "  Error tolerance = " << eps << "\n"
              << "  Mismatched elements = " << num_bad << " / " << size << "\n";
}

void init_random(float* data, int size, float low = -5.f, float high = 5.f) {
    std::mt19937 gen(123);  // 固定 seed，方便复现
    std::uniform_real_distribution<float> dist(low, high);

    for (int i = 0; i < size; ++i) {
        data[i] = dist(gen);
    }
}

int main(int argc, char **argv)
{
    int repeat_times = 10;
    double iStart, iElaps;
    int size = 1 << 7;
    float total_time;

    size_t bytes = size * sizeof(float);
    float *h_src = (float *)malloc(bytes);
    float *h_dst = (float *)malloc(bytes);
    float *kernel_dst = (float *)malloc(bytes);

    // initialize the array
    init_random(h_src, size);

    // device memory
    float *d_src = nullptr, *d_dst = nullptr;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_src), bytes));
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_dst), bytes));

    // copy H -> D
    CHECK(cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice));

    // cpu reduction
    iStart = seconds();
    cpu_softmax(h_src, h_dst, size);
    iElaps = seconds() - iStart;
    std::cout << GREEN << "[host]: elapsed = " << iElaps * 1000 << " ms " << RESET << std::endl << std::endl;

    // gpu warp reduce    
    CHECK(cudaMemset(d_dst, 0, bytes));
    total_time = TIME_RECORD(repeat_times, ([&]{iSoftmax_WarpReduce(d_src, d_dst, size);}));
    std::cout << GREEN << std::endl  << __FILE__ << ":" << __LINE__ << 
    " [device warp reduce]: elapsed = " << total_time / repeat_times << " ms " << RESET << std::endl;
    memset(kernel_dst, 0, bytes);
    CHECK(cudaMemcpy(kernel_dst, d_dst, bytes, cudaMemcpyDeviceToHost));
    verifyResult(h_dst, kernel_dst, size, 1e-3);  

    // gpu fusion version
    CHECK(cudaMemset(d_dst, 0, bytes));
    total_time = TIME_RECORD(repeat_times, ([&]{iSoftmax_WarpReduce_fusion(d_src, d_dst, size);}));
    std::cout << GREEN << std::endl  << __FILE__ << ":" << __LINE__ << 
    " [device warp reduce fusion]: elapsed = " << total_time / repeat_times << " ms " << RESET << std::endl;
    memset(kernel_dst, 0, bytes);
    CHECK(cudaMemcpy(kernel_dst, d_dst, bytes, cudaMemcpyDeviceToHost));
    verifyResult(h_dst, kernel_dst, size, 1e-3);  

    // gpu sgemv softmax
    CHECK(cudaMemset(d_dst, 0, bytes));
    total_time = TIME_RECORD(repeat_times, ([&]{iSgemv_softmax(d_src, d_dst, 1, size);}));
    std::cout << GREEN << std::endl  << __FILE__ << ":" << __LINE__ << 
    " [device sgemv softmax]: elapsed = " << total_time / repeat_times << " ms " << RESET << std::endl;
    memset(kernel_dst, 0, bytes);
    CHECK(cudaMemcpy(kernel_dst, d_dst, bytes, cudaMemcpyDeviceToHost));
    verifyResult(h_dst, kernel_dst, size, 1e-3);  

    
    free(h_src);
    free(h_dst);
    free(kernel_dst);
    cudaFree(d_src);
    cudaFree(d_dst);

    return 1;
}


// 更多softmax优化参考：https://zhuanlan.zhihu.com/p/341059988