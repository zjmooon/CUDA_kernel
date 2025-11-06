#include <iostream>
#include <cuda_runtime.h>
#include "../common.h"
#include <iomanip>

#define BLOCK_SIZE 32

// CPU reference: C = A * B
void sgemm_cpu(int M, int N, int K,
               const float* A, int lda,
               const float* B, int ldb,
               float* C, int ldc)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                // row-major: A[i*lda + k], B[k*ldb + j]
                sum += A[i*lda + k] * B[k*ldb + j];
            }
            C[i*ldc + j] = sum;
        }
    }
}

// Naive per-element kernel
// Each thread computes one C(i,j)
__global__ void kSgemmNaive(int M, int N, int K,
                        const float* __restrict__ A, int lda,
                        const float* __restrict__ B, int ldb,
                        float* C, int ldc)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * lda + k] * B[k * ldb + col];
        }
        C[row * ldc + col] = sum;
    }
}
void iSgemmNaive(int M, int N, int K,
                        const float* __restrict__ A, int lda,
                        const float* __restrict__ B, int ldb,
                        float* C, int ldc)
{
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));

    kSgemmNaive<<<gridSize, blockSize>>>(M, N, K, A, lda, B, ldb, C, ldc);
}


/* template <int TILE_DIM>
__global__ void kSgemmBlockTiled(int M, int N, int K,
                        const float* __restrict__ A, int lda,
                        const float* __restrict__ B, int ldb,
                        float* C, int ldc)
{
    // thread coords within block
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // global row/col this thread will compute
    int cx = blockIdx.x * TILE_DIM + tx;
    int cy = blockIdx.y * TILE_DIM + ty;

    // shared tiles
    __shared__ float sA[TILE_DIM][TILE_DIM];
    __shared__ float sB[TILE_DIM][TILE_DIM];

    float acc = 0.0f;

    # pragma unroll
    for (int k = 0; k < K; k += TILE_DIM) {
        int ax = k + tx;
        int by = k + ty; 

        // 简单的三目运算符可由nvcc编译为selp指令，性能较if语句好
        // 全局内存load合并访存，共享内存store不Bank conflict
        sA[ty][tx] = (cy < M && ax < K) ? A[cy * lda + ax] : 0.0f;
        sB[ty][tx] = (by < K && cx < N) ? B[by * ldb + cx] : 0.0f;

        __syncthreads();

        // Multiply-accumulate over the tile
        #pragma unroll
        for (int k_inner = 0; k_inner < TILE_DIM; ++k_inner) {
            acc += sA[ty][k_inner] * sB[k_inner][tx];
            // sB[][] load不存在Bank conflict
        }
        __syncthreads();
    }

    if (cy < M && cx < N) {
        C[cy * ldc + cx] = acc;
    }
}
void iSgemmBlockTiled(int M, int N, int K,
                        const float* __restrict__ A, int lda,
                        const float* __restrict__ B, int ldb,
                        float* C, int ldc)
{
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));

    kSgemmBlockTiled<32><<<gridSize, blockSize>>>(M, N, K, A, lda, B, ldb, C, ldc);
}
 */


template <int TILE_DIM>
__global__ void kSgemmBlockTiled(int M, int N, int K,
                        const float* __restrict__ A, int lda,
                        const float* __restrict__ B, int ldb,
                        float* C, int ldc) 
{
    const int gx = threadIdx.x + blockDim.x * blockIdx.x; 
    const int gy = threadIdx.y + blockDim.y * blockIdx.y; 

    if (gx >= N || gy >= M) return;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];
    /* __shared__ float As[TILE_DIM * TILE_DIM];
    __shared__ float Bs[TILE_DIM * TILE_DIM]; */

    A = &A[lda * TILE_DIM * blockIdx.y];
    B = &B[TILE_DIM * blockIdx.x];
    float acc = 0.0f;

    # pragma unroll
    for (int k = 0; k < K; k += TILE_DIM) {
        As[ty][tx] = A[ty * lda + tx];
        Bs[ty][tx] = B[ty * ldb + tx];
        /* As[ty * TILE_DIM + tx] = A[ty * lda + tx];
        Bs[ty * TILE_DIM + tx] = B[ty * ldb + tx];  */      
        __syncthreads();

        A += TILE_DIM;
        B += TILE_DIM * ldb;

        # pragma unroll
        for (int k_inner = 0; k_inner < TILE_DIM; k_inner++) {
            acc += As[ty][k_inner] * Bs[k_inner][tx];
            // Bs load不存在Bank conflict
            // acc += As[ty * TILE_DIM + k_inner] * Bs[k_inner * TILE_DIM + tx];
        }
        __syncthreads();
    }

    C = &C[ldc * TILE_DIM * blockIdx.y + TILE_DIM * blockIdx.x];
    C[ty * ldc + tx] = acc;   // 不是 C[gy * ldc + gx] = acc, 因为C首地址已经移动到局部了
}
void iSgemmBlockTiled(int M, int N, int K,
                        const float* __restrict__ A, int lda,
                        const float* __restrict__ B, int ldb,
                        float* C, int ldc)
{
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));

    kSgemmBlockTiled<32><<<gridSize, blockSize>>>(M, N, K, A, lda, B, ldb, C, ldc);
}


/* 
* 在block tile的基础上使用float4向量化优化。
* Using vectorized loads reduces the total number of instructions, reduces latency, and improves bandwidth utilization.
* 线程布局在block维度除以4。
* https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/
*/
template <int TILE_DIM>
__global__ void kSgemmBlockTiled_float4(int M, int N, int K,
                        const float* __restrict__ A, int lda,
                        const float* __restrict__ B, int ldb,
                        float* C, int ldc) 
{
    const int gx = threadIdx.x + blockDim.x * blockIdx.x; 
    const int gy = threadIdx.y + blockDim.y * blockIdx.y; 

    if (4 * gx >= N || gy >= M) return;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    A = &A[lda * TILE_DIM * blockIdx.y];
    B = &B[TILE_DIM * blockIdx.x];
    float4 acc4 = {0.0f, 0.0f, 0.0f, 0.0f};
    // float4 a4, b4;
    

    # pragma unroll
    for (int k = 0; k < K; k += TILE_DIM) {
        reinterpret_cast<float4*>(&(As[ty][tx * 4]))[0] = reinterpret_cast<const float4*>(A + ty * lda)[tx];
        reinterpret_cast<float4*>(&(Bs[ty][tx * 4]))[0] = reinterpret_cast<const float4*>(B + ty * ldb)[tx];
        // 虽然tx因为线程布局的变动变为原来的1/4，但不要修改A,B相关的tx索引。因为A B是输入，需要保留原始索引做 float --> float4
        /* a4 = reinterpret_cast<const float4*>(A + ty * lda)[tx]; // 不是(A)[ty * lda + tx]
        As[ty][tx * 4    ] = a4.x;
        As[ty][tx * 4 + 1] = a4.y;
        As[ty][tx * 4 + 2] = a4.z;
        As[ty][tx * 4 + 3] = a4.w;
        b4 = reinterpret_cast<const float4*>(B + ty * ldb)[tx];
        Bs[ty][tx * 4    ] = b4.x;
        Bs[ty][tx * 4 + 1] = b4.y;
        Bs[ty][tx * 4 + 2] = b4.z;
        Bs[ty][tx * 4 + 3] = b4.w; */
          
        __syncthreads();

        A += TILE_DIM;
        B += TILE_DIM * ldb;

        # pragma unroll
        for (int k_inner = 0; k_inner < TILE_DIM; k_inner++) {
            acc4.x += As[ty][k_inner] * Bs[k_inner][tx * 4    ];
            acc4.y += As[ty][k_inner] * Bs[k_inner][tx * 4 + 1];
            acc4.z += As[ty][k_inner] * Bs[k_inner][tx * 4 + 2];
            acc4.w += As[ty][k_inner] * Bs[k_inner][tx * 4 + 3];
        }
        __syncthreads();
    }

    C = &C[ldc * TILE_DIM * blockIdx.y + TILE_DIM * blockIdx.x];
    C[ty * ldc + tx * 4    ] = acc4.x;
    C[ty * ldc + tx * 4 + 1] = acc4.y;
    C[ty * ldc + tx * 4 + 2] = acc4.z;
    C[ty * ldc + tx * 4 + 3] = acc4.w;
}
void iSgemmBlockTiled_float4(int M, int N, int K,
                        const float* __restrict__ A, int lda,
                        const float* __restrict__ B, int ldb,
                        float* C, int ldc)
{
    dim3 blockSize(CEIL(BLOCK_SIZE, 4), BLOCK_SIZE); // (8, 32)，适应float4的线程布局
    dim3 gridSize(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));

    kSgemmBlockTiled_float4<32><<<gridSize, blockSize>>>(M, N, K, A, lda, B, ldb, C, ldc);
}


/* 
* thread tile版本与block tile版本相比，区别在于tt中一个线程会负责C中THREAD_TILE*THREAD_TILE个数据。计算访存比会更高，提高线程利用率
* 所以需要更改的地方有： 
* TILE_DIM ---> BLOCK_TILE * THREAD_TILE
* 两层循环将数据搬运到shared memory和register memory
* tx ---> tx + x * BLOCK_TILE 
* ty ---> ty + y * BLOCK_TILE
*/
template <const int BLOCK_TILE, const int THREAD_TILE>
__global__ void kSgemmThreadTiled(int M, int N, int K,
                        const float* __restrict__ A, int lda,
                        const float* __restrict__ B, int ldb,
                        float* C, int ldc)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    constexpr int TILE_DIM = BLOCK_TILE * THREAD_TILE; // 数据布局
    
    __shared__ float As[TILE_DIM][TILE_DIM]; // BLOCK_TILE*BLOCK_TILE --> BLOCK_TILE*THREAD_TILE*BLOCK_TILE*THREAD_TILE
    __shared__ float Bs[TILE_DIM][TILE_DIM]; // 因为线程布局不在与数据布局相同，一个线程现在要处理THREAD_TILE*THREAD_TILE倍数据

    A = &A[blockIdx.y * TILE_DIM * lda]; 
    B = &B[blockIdx.x * TILE_DIM];       
    float acc[THREAD_TILE][THREAD_TILE] = {0.0f}; // 一个线程处理THREAD_TILE * THREAD_TILE的数据

    #pragma unroll
    for (int k = 0; k < K; k += TILE_DIM) 
    {
        // 从global memory加载A到shared memory
        #pragma unroll
        for (int i = 0; i < THREAD_TILE; i++) 
        {
            #pragma unroll
            for (int j = 0; j < THREAD_TILE; j++) 
            {
                As[ty + i * BLOCK_TILE][tx + j * BLOCK_TILE] = A[(ty + BLOCK_TILE * i) * lda + tx + j * BLOCK_TILE];
                Bs[ty + i * BLOCK_TILE][tx + j * BLOCK_TILE] = B[(ty + BLOCK_TILE * i) * ldb + tx + j * BLOCK_TILE];
            }
        }
        __syncthreads();

        // 移动到下一个K维度tile
        A += TILE_DIM; //
        B += TILE_DIM * ldb;
        
        #pragma unroll
        for (int i = 0; i < THREAD_TILE; i++) 
        {
            #pragma unroll
            for (int j = 0; j < THREAD_TILE; j++) 
            {
                #pragma unroll
                for (int k_inner = 0; k_inner < TILE_DIM; k_inner++) 
                { 
                    acc[i][j] += As[ty + i * BLOCK_TILE][k_inner] * Bs[k_inner][tx + j * BLOCK_TILE]; 
                }
            }
        }
        __syncthreads();
    }
    
    C = &C[blockIdx.y * TILE_DIM * ldc + blockIdx.x * TILE_DIM]; 
    // 写回结果到global memory
    #pragma unroll
    for (int i = 0; i < THREAD_TILE; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE; j++) {
            C[(ty + i * BLOCK_TILE) * ldc + tx + j * BLOCK_TILE] = acc[i][j]; 
        }
    }
}
void iSgemmThreadTiled(int M, int N, int K,
                        const float* __restrict__ A, int lda,
                        const float* __restrict__ B, int ldb,
                        float* C, int ldc)
{
    const int block_tile = 16;
    const int thread_tile = 4;
    // 3060:每个线程块的shared memory最大值：48 KB
    // 共享内存使用：As[64*64] + Bs[64*64] = 16KB + 16KB = 32KB < 48KB
    // 4080S:每个线程块的shared memory最大值：99 KB
    // 现在的核函数设计不能最大化利用shared memory，是一个性能优化点

    dim3 blockSize(block_tile, block_tile);
    dim3 gridSize(CEIL(N, block_tile * thread_tile), CEIL(M, block_tile * thread_tile));
    kSgemmThreadTiled<block_tile, thread_tile><<<gridSize, blockSize>>>(M, N, K, A, lda, B, ldb, C, ldc);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
}



// initialize matrix
void rand_matrix(float* mat, int rows, int cols, int ld) {
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            mat[i*ld + j] = (float(rand()) / RAND_MAX) * 2.0f - 1.0f;
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
            // std::cout << "host " << host[i] << ", kernel " << kernel[i] << std::endl;
        }

        if (diff > max_abs_err)
            max_abs_err = diff;
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



int main(int argc, char** argv)
{
    int repeat_times = 5;
    double iStart, iElaps;
    int M = 1024 * 2;
    int N = 1024 * 2;
    int K = 1024 * 2;
    float total_time;

    // Leading dims (row-major)行主序
    int lda = K;
    int ldb = N;
    int ldc = N;
    size_t sizeA = (size_t)M * lda;
    size_t sizeB = (size_t)K * ldb;
    size_t sizeC = (size_t)M * ldc;

    // allocate host memory
    float* hA = (float*)malloc(sizeA * sizeof(float));
    float* hB = (float*)malloc(sizeB * sizeof(float));
    float* hC = (float*)malloc(sizeC * sizeof(float));
    float* kernel_C = (float*)malloc(sizeC * sizeof(float));

    rand_matrix(hA, M, K, lda); // (M行K列)
    rand_matrix(hB, K, N, ldb); // (K行N列)
    memset(hC, 0, sizeC * sizeof(float)); // (M行N列)
    memset(kernel_C, 0, sizeC * sizeof(float)); // (M行N列)
    // rand_matrix(hC, M, N, ldc); 


    // allocate deveice memory
    float *dA, *dB, *dC;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&dA), sizeA * sizeof(float)));
    CHECK(cudaMalloc(reinterpret_cast<void **>(&dB), sizeB * sizeof(float)));
    CHECK(cudaMalloc(reinterpret_cast<void **>(&dC), sizeC * sizeof(float)));

    // copy H -> D
    CHECK(cudaMemcpy(dA, hA, sizeA * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, hB, sizeB * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dC, hC, sizeC * sizeof(float), cudaMemcpyHostToDevice));

    // CPU reference (for correctness)
    iStart = seconds();
    sgemm_cpu(M, N, K, hA, lda, hB, ldb, hC, ldc);
    iElaps = seconds() - iStart;
    std::cout << GREEN << "[host]: elapsed = " << iElaps * 1000 << " ms, " << RESET << std::endl << std::endl;

    // navie
    CHECK(cudaMemset(dC, 0, sizeC * sizeof(float)));
    total_time = TIME_RECORD(repeat_times, ([&]{iSgemmNaive(M, N, K, dA, lda, dB, ldb, dC, ldc);}));
    std::cout << GREEN << std::endl << "[device navie]: elapsed = " << total_time / repeat_times << " ms " << RESET << std::endl;
    memset(kernel_C, 0, sizeC * sizeof(float));
    CHECK(cudaMemcpy(kernel_C, dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost));
    verifyResult(hC, kernel_C, M * N, 1e-3);  

    CHECK(cudaMemset(dC, 0, sizeC * sizeof(float)));
    total_time = TIME_RECORD(repeat_times, ([&]{iSgemmBlockTiled(M, N, K, dA, lda, dB, ldb, dC, ldc);}));
    std::cout << GREEN << std::endl << "[device block tile]: elapsed = " << total_time / repeat_times << " ms " << RESET << std::endl;
    memset(kernel_C, 0, sizeC * sizeof(float));
    CHECK(cudaMemcpy(kernel_C, dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost));
    verifyResult(hC, kernel_C, M * N, 1e-3);  

    /* CHECK(cudaMemset(dC, 0, sizeC * sizeof(float)));
    total_time = TIME_RECORD(repeat_times, ([&]{iSgemmBlockTiled2(M, N, K, dA, lda, dB, ldb, dC, ldc);}));
    std::cout << GREEN << std::endl << "[device block tile (another version)]: elapsed = " << total_time / repeat_times << " ms " << RESET << std::endl;
    memset(kernel_C, 0, sizeC * sizeof(float));
    CHECK(cudaMemcpy(kernel_C, dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost));
    verifyResult(hC, kernel_C, M * N, 1e-3); */  

    CHECK(cudaMemset(dC, 0, sizeC * sizeof(float)));
    total_time = TIME_RECORD(repeat_times, ([&]{iSgemmThreadTiled(M, N, K, dA, lda, dB, ldb, dC, ldc);}));
    std::cout << GREEN << std::endl << "[device thread tile]: elapsed = " << total_time / repeat_times << " ms " << RESET << std::endl;
    memset(kernel_C, 0, sizeC * sizeof(float));
    CHECK(cudaMemcpy(kernel_C, dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost));
    verifyResult(hC, kernel_C, M * N, 1e-3);  

    CHECK(cudaMemset(dC, 0, sizeC * sizeof(float)));
    total_time = TIME_RECORD(repeat_times, ([&]{iSgemmBlockTiled_float4(M, N, K, dA, lda, dB, ldb, dC, ldc);}));
    std::cout << GREEN << std::endl << "[device block tile float4]: elapsed = " << total_time / repeat_times << " ms " << RESET << std::endl;
    memset(kernel_C, 0, sizeC * sizeof(float));
    CHECK(cudaMemcpy(kernel_C, dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost));
    verifyResult(hC, kernel_C, M * N, 1e-3);  

    // cleanup
    cudaFree(dA); 
    cudaFree(dB); 
    cudaFree(dC);
    free(hA); 
    free(hB); 
    free(hC); 
    free(kernel_C);

    return 1;
}
