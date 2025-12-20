#include "../common.h"
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>


__global__ void kVectorAdd_Naive(const float* __restrict__ A, const float* __restrict__ B, 
    float* __restrict__ C, int N) 
{
    int g_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (g_id < N) {
        C[g_id] = A[g_id] + B[g_id];
    }
}
void iVectorAdd_Naive(const float* A, const float* B, float* C, int N) 
{
    // 线程布局：每个线程处理一个数据(float)
    int blockSize = 512;
    int gridSize = CEIL(N, blockSize); 

    kVectorAdd_Naive<<<gridSize, blockSize>>>(A, B, C, N);
    CHECK(cudaDeviceSynchronize());
}



__global__ void kVectorAdd_Float4(const float* __restrict__ A, const float* __restrict__ B, 
    float* __restrict__ C, int N) 
{
    int g_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (g_id * 4 >= N) return;

    const float4* float4_A = reinterpret_cast<const float4*>(A);
    const float4* float4_B = reinterpret_cast<const float4*>(B);
    float4* float4_C = reinterpret_cast<float4*>(C);

    float4_C[g_id] = make_float4(   (float4_A[g_id].x + float4_B[g_id].x),
                                    (float4_A[g_id].y + float4_B[g_id].y),
                                    (float4_A[g_id].z + float4_B[g_id].z),
                                    (float4_A[g_id].w + float4_B[g_id].w)    );

}
void iVectorAdd_Float4(const float* A, const float* B, float* C, int N) 
{
    int blockSize = 512;
    int gridSize = CEIL(CEIL(N, 4), blockSize); 

    kVectorAdd_Float4<<<gridSize, blockSize>>>(A, B, C, N);
    CHECK(cudaDeviceSynchronize());
}



__global__ void kVectorAdd_GridStride(const float* __restrict__ A, const float* __restrict__ B, 
    float* __restrict__ C, int N) 
{
    int g_id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = g_id; i < N; i += stride) {
        C[i] = A[i] + B[i];
    }
}
void iVectorAdd_GridStride(const float* A, const float* B, float* C, int N) 
{
    int blockSize = 512;
    // Grid Size Calculation (针对 Grid-Stride Loop 的优化)
    // 不需要 (N + blockSize - 1) / blockSize 这么大的 Grid，通过(int i = g_id; i < N; i += stride)会自动分配
    // 只需要生成足够填满 GPU 所有 SM 的线程即可
    // 这里的逻辑是：获取 SM 数量，每个 SM 跑 32 个 Block (经验值，保证占有率)
    int numSMs;
    int dev;
    CHECK(cudaGetDevice(&dev));
    CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, dev));
    
    int gridSize = 32 * numSMs; 
    
    // printf("Launching kernel with Grid Size: %d, Block Size: %d\n", gridSize, blockSize);

    kVectorAdd_GridStride<<<gridSize, blockSize>>>(A, B, C, N);
    CHECK(cudaDeviceSynchronize());
}



__global__ void kVectorAdd_gridStride_Float4(const float* __restrict__ A, const float* __restrict__ B, 
    float* __restrict__ C, int N) 
{
    const float4* f4_A = reinterpret_cast<const float4*>(A);
    const float4* f4_B = reinterpret_cast<const float4*>(B);
    float4* C_vec = reinterpret_cast<float4*>(C);

    int g_id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    // 使用float4，以4个数据为单位/迭代
    const int f4_N = N / 4;

    for (int i = g_id; i < f4_N; i += stride) 
    {
        float4 f4_a = f4_A[i]; // 1条指令加载 128 bit
        float4 f4_b = f4_B[i]; // 1条指令加载 128 bit

        C_vec[i] = make_float4(f4_a.x + f4_b.x, 
                                  f4_a.y + f4_b.y, 
                                  f4_a.z + f4_b.z, 
                                  f4_a.w + f4_b.w); // 一条指令写入 128 bit
    }
        
    int tail = f4_N * 4 + g_id;
    if (tail < N) {
        C[tail] = A[tail] + B[tail];
    }
}
void iVectorAdd_gridStride_Float4(const float* A, const float* B, float* C, int N) 
{
    int blockSize = 512;
    // Grid Size Calculation (针对 Grid-Stride Loop 的优化)
    // 不需要 (N + blockSize - 1) / blockSize 这么大的 Grid
    // 只需要生成足够填满 GPU 所有 SM 的线程即可。
    // 这里的逻辑是：获取 SM 数量，每个 SM 跑 32 个 Block (经验值，保证占有率)
    int numSMs;
    int dev;
    CHECK(cudaGetDevice(&dev));
    CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, dev));
    int gridSize = 32 * numSMs; 
    
    kVectorAdd_gridStride_Float4<<<gridSize, blockSize>>>(A, B, C, N);
    CHECK(cudaDeviceSynchronize());
}



void vectorAdd_cpu(const float* h_A, const float* h_B, float* h_Ref, int N)
{
    for(int i = 0; i < N; i++) {
        h_Ref[i] = h_A[i] + h_B[i];
    }
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

}

int main(int argc, char **argv) {
    int n = 1 << 27; 
    int repeat_times = 10;
    double iStart, iElaps;
    size_t nBytes = n * sizeof(float);
    std::cout << "Vector size: " << n << " elements (" <<  nBytes / (1024.0 * 1024.0) << "MB)" << std::endl;
    float total_time;

    float *h_A, *h_B, *h_C, *h_Ref;
    h_A   = (float *)malloc(nBytes);
    h_B   = (float *)malloc(nBytes);
    h_C   = (float *)malloc(nBytes); 
    h_Ref = (float *)malloc(nBytes); 

    // init 
    for (int i = 0; i < n; i++) {
        h_A[i] = (float(rand()) / RAND_MAX) * 2.0f - 1.0f;
        h_B[i] = (float(rand()) / RAND_MAX) * 2.0f - 1.0f;
    }

    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), nBytes));
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), nBytes));
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C), nBytes));

    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));

    // cpu reference
    iStart = seconds();
    vectorAdd_cpu(h_A, h_B, h_Ref, n);
    iElaps = seconds() - iStart;
    std::cout << GREEN << "[host]: elapsed = " << iElaps * 1000 << " ms "<< RESET << std::endl << std::endl;

    // gpu naive
    CHECK(cudaMemset(d_C, 0, nBytes));
    total_time = TIME_RECORD(repeat_times, ([&]{iVectorAdd_Naive(d_A, d_B, d_C, n);}));
    std::cout << GREEN << std::endl  << __FILE__ << ":" << __LINE__ << 
    " [device naive]: elapsed = " << total_time / repeat_times << " ms " << RESET << std::endl;
    memset(h_C, 0, nBytes);
    CHECK(cudaMemcpy(h_C, d_C, nBytes, cudaMemcpyDeviceToHost));
    verifyResult(h_Ref, h_C, n);  

    // gpu naive
    CHECK(cudaMemset(d_C, 0, nBytes));
    total_time = TIME_RECORD(repeat_times, ([&]{iVectorAdd_Float4(d_A, d_B, d_C, n);}));
    std::cout << GREEN << std::endl  << __FILE__ << ":" << __LINE__ << 
    " [device float4]: elapsed = " << total_time / repeat_times << " ms " << RESET << std::endl;
    memset(h_C, 0, nBytes);
    CHECK(cudaMemcpy(h_C, d_C, nBytes, cudaMemcpyDeviceToHost));
    verifyResult(h_Ref, h_C, n);  


    // gpu gride stride
    CHECK(cudaMemset(d_C, 0, nBytes));
    total_time = TIME_RECORD(repeat_times, ([&]{iVectorAdd_GridStride(d_A, d_B, d_C, n);}));
    std::cout << GREEN << std::endl  << __FILE__ << ":" << __LINE__ << 
    " [device gridStride]: elapsed = " << total_time / repeat_times << " ms " << RESET << std::endl;
    memset(h_C, 0, nBytes);
    CHECK(cudaMemcpy(h_C, d_C, nBytes, cudaMemcpyDeviceToHost));
    verifyResult(h_Ref, h_C, n);  

    // gpu gride stride float4
    CHECK(cudaMemset(d_C, 0, nBytes));
    total_time = TIME_RECORD(repeat_times, ([&]{iVectorAdd_gridStride_Float4(d_A, d_B, d_C, n);}));
    std::cout << GREEN << std::endl  << __FILE__ << ":" << __LINE__ << 
    " [device gridStride float4]: elapsed = " << total_time / repeat_times << " ms " << RESET << std::endl;
    memset(h_C, 0, nBytes);
    CHECK(cudaMemcpy(h_C, d_C, nBytes, cudaMemcpyDeviceToHost));
    verifyResult(h_Ref, h_C, n);  

    // cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_Ref);

    return 0;
}
