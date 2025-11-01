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

    __shared__ float sA[TILE_DIM][TILE_DIM];
    __shared__ float sB[TILE_DIM][TILE_DIM];
    /* __shared__ float sA[TILE_DIM * TILE_DIM];
    __shared__ float sB[TILE_DIM * TILE_DIM]; */

    A = &A[lda * TILE_DIM * blockIdx.y];
    B = &B[TILE_DIM * blockIdx.x];
    C = &C[ldc * TILE_DIM * blockIdx.y + TILE_DIM * blockIdx.x];

    float acc = 0.0f;

    # pragma unroll
    for (int k = 0; k < K; k += TILE_DIM) {
        sA[ty][tx] = A[ty * lda + tx];
        sB[ty][tx] = B[ty * ldb + tx];
        /* sA[ty * TILE_DIM + tx] = A[ty * lda + tx];
        sB[ty * TILE_DIM + tx] = B[ty * ldb + tx];  */      
        __syncthreads();

        A += TILE_DIM;
        B += TILE_DIM * ldb;

        # pragma unroll
        for (int i = 0; i < TILE_DIM; i++) {
            acc += sA[ty][i] * sB[i][tx];
            // sB load不存在Bank conflict
            // acc += sA[ty * TILE_DIM + i] * sB[i * TILE_DIM + tx];
        }
        __syncthreads();
    }

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



// Block-tiled kernel with shared memory
// Row-major assumed. TILE_DIM should divide or handle boundaries.
template <int TILE_DIM>
__global__ void kSgemmBlockTiled_V2(int M, int N, int K,
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
    for (int k_tile = 0; k_tile < K; k_tile += TILE_DIM) {
        int ax = k_tile + tx;
        int by = k_tile + ty; 

        // 简单的三目运算符可由nvcc编译为selp指令，性能较if语句好
        // 全局内存load合并访存，共享内存store不Bank conflict
        sA[ty][tx] = (cy < M && ax < K) ? A[cy * lda + ax] : 0.0f;
        sB[ty][tx] = (by < K && cx < N) ? B[by * ldb + cx] : 0.0f;

        __syncthreads();

        // Multiply-accumulate over the tile
        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            acc += sA[ty][k] * sB[k][tx];
            // sB[][] load存在Bank conflict (padding解决)
        }
        __syncthreads();
    }

    if (cy < M && cx < N) {
        C[cy * ldc + cx] = acc;
    }
}
void iSgemmBlockTiled_V2(int M, int N, int K,
                        const float* __restrict__ A, int lda,
                        const float* __restrict__ B, int ldb,
                        float* C, int ldc)
{
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));

    kSgemmBlockTiled_V2<32><<<gridSize, blockSize>>>(M, N, K, A, lda, B, ldb, C, ldc);
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
    total_time = TIME_RECORD(repeat_times, ([&]{iSgemmNaive(M, N, K, dA, lda, dB, ldb, dC, ldc);}));
    std::cout << GREEN << std::endl << "[device navie]: elapsed = " << total_time / repeat_times << " ms " << RESET << std::endl;
    CHECK(cudaMemcpy(hC, dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost));
    // verifyResult(hC, kernel_C, M * N, 1e-3);  

    total_time = TIME_RECORD(repeat_times, ([&]{iSgemmBlockTiled(M, N, K, dA, lda, dB, ldb, dC, ldc);}));
    std::cout << GREEN << std::endl << "[device block tile]: elapsed = " << total_time / repeat_times << " ms " << RESET << std::endl;
    CHECK(cudaMemcpy(kernel_C, dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost));
    verifyResult(hC, kernel_C, M * N, 1e-3);  

    total_time = TIME_RECORD(repeat_times, ([&]{iSgemmBlockTiled_V2(M, N, K, dA, lda, dB, ldb, dC, ldc);}));
    std::cout << GREEN << std::endl << "[device block tile (another version)]: elapsed = " << total_time / repeat_times << " ms " << RESET << std::endl;
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