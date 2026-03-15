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
__global__ 
void kSgemv_softmax(const float* __restrict__ input, float* __restrict__ output, int x_num) 
{
    const int tx = threadIdx.x;
    const int bx = blockIdx.x;

    /* __shared__ float max_val_shared; // Block内共享变量
    __shared__ float sum_val_shared; */

    // max
    // 先求出每个线程负责数据的max_val
    float max_val = -FLT_MAX;
    for (int i = tx; i < x_num; i += warpSize) {
        max_val = fmaxf(input[bx * x_num + i], max_val);
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
    for (int i = tx; i < x_num; i += warpSize) {
        /* sum_val += expf(input[bx * x_num + i] - max_val_shared); */
        sum_val += expf(input[bx * x_num + i] - max_val);
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum_val += __shfl_xor_sync(0XFFFFFFFF, sum_val, offset);
    }
    // __shfl_xor_sync: 已经求得sum_val(每个线程中)
    /* if (tx == 0) {
        sum_val_shared = sum_val;
    } */

    // 对一行数据的所有数据做softmax
    for (int i = tx; i < x_num; i += warpSize) {
        output[bx * x_num + i] = expf(input[bx * x_num + i] - max_val) / sum_val;
    }

}
void iSgemv_softmax(const float* __restrict__ input, float* __restrict__ output, int y_num, int x_num) {
    int blockSize(32); // block的大小设置为warpSize
    int gridSize(y_num); // 每个block负责一行.每个 Block 只有一个线程束(Warp).在处理列数K较小的矩阵时非常高效

    kSgemv_softmax<<<gridSize, blockSize>>>(input, output, x_num);
}

/*
* softmax(xi) = exp(xi - M) / exp(xi - M).sum()
* flash attention中常用的online softmax版本，在线计算max和sum，减少内存访问
* https://zhuanlan.zhihu.com/p/5078640012
*/ 
__global__
void kOnline_softmax(const float* __restrict__ input, float* __restrict__ output,    
                    const int num_rows, const int num_cols, int Bc) 
{
    int row_idx = blockIdx.x;
    int tx = threadIdx.x;
    if (row_idx >= num_rows) return;

    int row_offset = row_idx * num_cols;

    float row_m_prev = -INFINITY;
    float row_l_prev = 0.0f;

    // 将每行划分为多个块，每个块包含 Bc 列
    int num_chunks = CEIL(num_cols, Bc);

    // 以下的设计方式并没有做到合并访存，主要是为了复现流式计算的思路   TODO: 合并访存优化
    // Phase 1: Online Reduction (流式分块计算最终的 m 和 l)    
    for (int j = 0; j < num_chunks; j++) {
        int start_col = j * Bc;
        int end_col = min(start_col + Bc, num_cols);

        float local_m = -INFINITY;
        float local_l = 0.0f;

        // 1. 计算当前块的最大值
        #pragma unroll
        for (int col = start_col; col < end_col; col++) {
            local_m = fmaxf(local_m, input[row_offset + col]);
        }
        // 2. 更新行最大值
        float row_m_new = fmaxf(row_m_prev, local_m);

        // 3. 计算当前块的局部指数和
        #pragma unroll
        for (int col = start_col; col < end_col; col++) {
            local_l += __expf(input[row_offset + col] - row_m_new);
        }

        // 4. 更新行指数和(根据新旧最大值的差，缩放旧的指数和，并加上当前块的指数和)
        float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + local_l;

        // 5. 更新全局的 m 和 l
        row_m_prev = row_m_new;
        row_l_prev = row_l_new;
    }
    /* // 如果后续需要反向传播，需要将每行的最终 m 和 l 保存下来，供后续使用
    if (m_out != nullptr) m_out[row_idx] = row_m_prev;
    if (l_out != nullptr) l_out[row_idx] = row_l_prev; */

    // Phase 2: 生成最终的 Softmax 概率矩阵, flash attention的精妙之处在于不需要显式地进行这一步骤
    #pragma unroll
    for (int col = tx; col < num_cols; col += blockDim.x) {
        output[row_offset + col] = __expf(input[row_offset + col] - row_m_prev) / row_l_prev;
    }
}
void iOnline_softmax(const float* __restrict__ input, float* __restrict__ output, int num_rows, int num_cols, int Bc) {
   
    dim3 blockSize(1);
    dim3 gridSize(num_rows); // 每个block负责一行
    
    kOnline_softmax<<<gridSize, blockSize>>>(input, output, num_rows, num_cols, Bc);
}


/*
* softmax(xi) = exp(xi - M) / exp(xi - M).sum()
* 以行为单位做softmax
*/ 
__global__ 
void kSgemv_online_softmax(const float* __restrict__ input, float* __restrict__ output, int num_cols) 
{
    const int tx = threadIdx.x;
    const int bx = blockIdx.x;

    // 先求出每个线程负责数据的max和sum
    float m = -FLT_MAX;
    float l = 0.0f;

    for (int i = tx; i < num_cols; i += warpSize) {
        float max_prev = m;
        float cur_input = input[bx * num_cols + i];
        m = fmaxf(cur_input, max_prev);
        l = __expf(max_prev - m) * l + __expf(cur_input - m);
    }

    // 分块流式得到一行的max和sum
    // warp内归约
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        float max_offset = __shfl_xor_sync(0XFFFFFFFF, m, offset);
        float l_offset = __shfl_xor_sync(0XFFFFFFFF, l, offset);
        float max_new = fmaxf(m, max_offset);
        
        l = __expf(m - max_new) * l + __expf(max_offset - max_new) * l_offset;
        m = max_new;
    } 

    /* // 如果后续需要反向传播，需要将每行的最终 m 和 l 保存下来，供后续使用
    if (m_out != nullptr) m_out[row_idx] = m;
    if (l_out != nullptr) l_out[row_idx] = l; */

    // 对一行数据的所有数据做softmax
    for (int i = tx; i < num_cols; i += warpSize) {
        output[bx * num_cols + i] = expf(input[bx * num_cols + i] - m) / l;
    }

}
void iSgemv_online_softmax(const float* __restrict__ input, float* __restrict__ output, int num_rows, int num_cols) {
    int blockSize(32); // block的大小设置为warpSize
    int gridSize(num_rows); // 每个block负责一行.每个 Block 只有一个线程束(Warp)

    kSgemv_online_softmax<<<gridSize, blockSize>>>(input, output, num_cols);
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
    int size = 1 << 9;
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

/*     // gpu fusion version (further verification is needed)
    CHECK(cudaMemset(d_dst, 0, bytes));
    total_time = TIME_RECORD(repeat_times, ([&]{iSoftmax_WarpReduce_fusion(d_src, d_dst, size);}));
    std::cout << GREEN << std::endl  << __FILE__ << ":" << __LINE__ << 
    " [device warp reduce fusion]: elapsed = " << total_time / repeat_times << " ms " << RESET << std::endl;
    memset(kernel_dst, 0, bytes);
    CHECK(cudaMemcpy(kernel_dst, d_dst, bytes, cudaMemcpyDeviceToHost));
    verifyResult(h_dst, kernel_dst, size, 1e-3);   */

    // gpu sgemv-like softmax
    CHECK(cudaMemset(d_dst, 0, bytes));
    total_time = TIME_RECORD(repeat_times, ([&]{iSgemv_softmax(d_src, d_dst, 1, size);})); // 以行为单位做softmax，为便于验证，令行数为1，列数为size
    std::cout << GREEN << std::endl  << __FILE__ << ":" << __LINE__ << 
    " [device sgemv-like softmax]: elapsed = " << total_time / repeat_times << " ms " << RESET << std::endl;
    memset(kernel_dst, 0, bytes);
    CHECK(cudaMemcpy(kernel_dst, d_dst, bytes, cudaMemcpyDeviceToHost));
    verifyResult(h_dst, kernel_dst, size, 1e-3);  

    // gpu online_softmax
    CHECK(cudaMemset(d_dst, 0, bytes));
    total_time = TIME_RECORD(repeat_times, ([&]{iOnline_softmax(d_src, d_dst, 1, size, 32);})); // 以行为单位做softmax，为便于验证，令行数为1，列数为size，分块大小为32
    std::cout << GREEN << std::endl  << __FILE__ << ":" << __LINE__ << 
    " [device online softmax]: elapsed = " << total_time / repeat_times << " ms " << RESET << std::endl;
    memset(kernel_dst, 0, bytes);
    CHECK(cudaMemcpy(kernel_dst, d_dst, bytes, cudaMemcpyDeviceToHost));
    verifyResult(h_dst, kernel_dst, size, 1e-3);  
    
    // gpu sgemv-like online_softmax
    CHECK(cudaMemset(d_dst, 0, bytes));
    total_time = TIME_RECORD(repeat_times, ([&]{iSgemv_online_softmax(d_src, d_dst, 1, size);})); // 以行为单位做softmax，为便于验证，令行数为1，列数为size
    std::cout << GREEN << std::endl  << __FILE__ << ":" << __LINE__ << 
    " [device sgemv-like online softmax]: elapsed = " << total_time / repeat_times << " ms " << RESET << std::endl;
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
