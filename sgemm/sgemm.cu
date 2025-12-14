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
    __shared__ float Bs[TILE_DIM][TILE_DIM]; // 因为线程布局不再与数据布局相同，一个线程现在要处理THREAD_TILE*THREAD_TILE倍数据

    A = &A[blockIdx.y * TILE_DIM * lda]; 
    B = &B[blockIdx.x * TILE_DIM];       
    float acc[THREAD_TILE][THREAD_TILE] = {0.0f}; // 一个线程处理THREAD_TILE * THREAD_TILE的数据

    #pragma unroll
    for (int k = 0; k < K; k += TILE_DIM) 
    {
        // 从global memory搬运到shared memory
        // 每个线程搬运THREAD_TILE*THREAD_TILE个数据，这些数据间隔为BLOCK_TILE(x,y方向都满足)
        #pragma unroll
        for (int i = 0; i < THREAD_TILE; i++) 
        {
            #pragma unroll
            for (int j = 0; j < THREAD_TILE; j++) 
            {
                // 会有BLOCK_TILE的跨度
                As[ty + i * BLOCK_TILE][tx + j * BLOCK_TILE] = A[(ty + BLOCK_TILE * i) * lda + tx + j * BLOCK_TILE];
                Bs[ty + i * BLOCK_TILE][tx + j * BLOCK_TILE] = B[(ty + BLOCK_TILE * i) * ldb + tx + j * BLOCK_TILE];
            }
        }
        __syncthreads();

        // 移动到下一个K维度tile
        A += TILE_DIM; //
        B += TILE_DIM * ldb;

        #pragma unroll
        for (int k_inner = 0; k_inner < TILE_DIM; k_inner++) { // 这个循环的位置可以在最外层或者最内层
            #pragma unroll
            for (int i = 0; i < THREAD_TILE; i++) {
                #pragma unroll
                for (int j = 0; j < THREAD_TILE; j++) {
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

// 比较常见的thread tile版本
// BM BN BK TM TN都属于数据布局的变量
template<const int BM,
         const int BN,
         const int BK,
         const int TM,
         const int TN>
__global__ void kSgemmThreadTiled_ref(const float* __restrict__ A, const float* __restrict__ B,
                        float* C, int M, int N, int K) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // 因为此核函数的线程布局blockSize是一维的，所以需要手动计算 threadIdx.x,threadIdx.y
    int block_dim_x = BN / TN;  // block中一行的thread数量 (128/8=16)
    int block_dim_y = BM / TM;  // block中一列的thread数量 (128/8=16)
    int thread_num = block_dim_x * block_dim_y;  // block中thread总量 (16*16=256)  ?blockDim.x

    // (threadTile操作(线程操作更多数据)，加倍Mul, 后续时填充+(0~Mul-1))
    int tx = (threadIdx.x % block_dim_x) * TN;  // threadtile左上角x坐标 ((0~255) % 16) * 8 --> (0~15) * 8 --> (0,8,16,...,120)  
    int ty = (threadIdx.x / block_dim_x) * TM;  // threadtile左上角y坐标 ((0~255) / 16) * 8 --> (0~15) * 8 --> (0,8,16,...,120)  

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    // 从global memory搬运到shared memory
    // 可以不利用threadTile操作，所以设计成一个线程搬运少数数据：BM/a_tile_stride(与TM*TN比较)
    // 手动计算线程在As中的x，y方向坐标
    int a_tile_y = threadIdx.x / BK;
    int a_tile_x = threadIdx.x % BK;
    // 目前为止，已经将线程映射到了 As 矩阵的上方一部分区域。如果线程不够多，它们只能覆盖矩阵的上半截。
    int a_tile_stride = thread_num / BK; 
    // threadIdx.x,thread_num是线程布局变量，BK是数据布局变量。
    // thread_num是线程布局变量block thread总量，BK是数据布局变量As矩阵的一行有多少列。
    // 物理意义：a_tile_stride是block所有线程一次性总共能搬运多少行。 

    int b_tile_y = threadIdx.x / BN; // 以block thread为单位划分Bs数据
    int b_tile_x = threadIdx.x % BN;
    int b_tile_stride = thread_num / BN; // threadIdx.x,thread_num是线程布局变量，BN是数据布局变量

    float accum[TM][TN] = {0.0f};
    float a_frag[TM] = {0.0f};
    float b_frag[TN] = {0.0f};
    
    #pragma unroll
    for (int k = 0; k < K; k += BK) {
        // 从global memory搬运到shared memory
        // 每个线程搬运 BM/a_tile_stride个数据，这些数据间隔为a_tile_stride
        #pragma unroll
        for (int i = 0; i < BM; i += a_tile_stride) { // 一共有BM行
            As[a_tile_y + i][a_tile_x] = A[(a_tile_y + i) * K + a_tile_x];
        }
        #pragma unroll
        for (int i = 0; i < BK; i += b_tile_stride) {
            Bs[b_tile_y + i][b_tile_x] = B[(b_tile_y + i) * N + b_tile_x];
        }
        __syncthreads();

        A += BK;
        B += BK * N;

        /* 
        // 内积，效率比较低。访问TM*TN*2次shared_memory
        #pragma unroll
        for (int y = 0; y < TM; y++) {
            #pragma unroll
            for (int x = 0; x < TN; x++) {
                #pragma unroll
                for (int k_inner = 0; k_inner < BK; k_inner++) {
                    // tx,ty:(0,8,16,...,120)
                    // y,x:(0~7)
                    // 对于线程[ty，tx]，它计算C中TM*TN个数据
                    // 对于As、Bs, 每一个数据的计算需要As的一行与Bs的一列(A一行的部分和B一列的部分)
                    // 计算一个accum[y][x]要访问As、Bs各TM*TN次，可以搬运shared memory数据到register memory中提高访问速度
                    accum[y][x] += As[ty + y][k_inner] * Bs[k_inner][tx + x];
                }
            }
        } */

        // 外积，效率好。访问TM+TN次shared_memory 
        #pragma unroll
        for (int k_inner = 0; k_inner < BK; k_inner++) {
            // 搬运shared memory数据到register memory中提高访问速度
            // As的一列的TM个数据 As[[BM][BK]
            // BM%TM==0
            #pragma unroll
            for (int y = 0; y < TM; y++) {
                a_frag[y] = As[ty + y][k_inner];
            }
            // Bs的一行的TN个数据 Bs[BK][BN]
            #pragma unroll
            for (int x = 0; x < TN; x++) {
                b_frag[x] = Bs[k_inner][tx + x];
            }
            #pragma unroll
            for (int y = 0; y < TM; y++) {
                #pragma unroll
                for (int x = 0; x < TN; x++)
                    accum[y][x] += a_frag[y] * b_frag[x];
            }
        }
       
        __syncthreads();
    }

    
    #pragma unroll
    for (int y = 0; y < TM; y++) {
        for (int x = 0; x < TN; x++) {
            C[(ty + y) * N + (tx + x)] = accum[y][x];
        }
    }
}
void iSgemmThreadTiled_ref(int M, int N, int K,
                        const float* __restrict__ A, int lda,
                        const float* __restrict__ B, int ldb,
                        float* C, int ldc)
{
    dim3 block(256);
    dim3 grid(CEIL(N, 128), CEIL(M, 128));

    // 模版参数的设置需要满足： (BM*BN)/(TM*TN) == block.x(一维)
    kSgemmThreadTiled_ref<128, 128, 8, 8, 8><<<grid, block>>>(A, B, C, M, N, K);
}



// 修改一下block布局的维度 耗时明显增大？
template<const int BM,
         const int BN,
         const int BK,
         const int TM,
         const int TN>
__global__ void kSgemmThreadTiled_ref_2d(const float* __restrict__ A, const float* __restrict__ B,
                        float* C, int M, int N, int K) {

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int thread_num = blockDim.y * blockDim.x; // 一个block中的thread数量

    // (threadTile操作(线程操作更多数据)，加倍Mul, 后续时填充+(0~Mul-1))
    int tx = threadIdx.x * TN;  // threadtile左上角x坐标 ((0~255) % 16) * 8 --> (0~15) * 8 --> (0,8,16,...,120)  
    int ty = threadIdx.y * TM;  // threadtile左上角y坐标 ((0~255) / 16) * 8 --> (0~15) * 8 --> (0,8,16,...,120)  
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    // 从global memory搬运到shared memory
    // 可以不利用threadTile操作，所以设计成一个线程搬运少数数据：BM/a_tile_stride(与TM*TN比较)
    int a_tile_y = tid / BK;
    int a_tile_x = tid % BK;
    int a_tile_stride = thread_num / BK;
    int b_tile_y = tid / BN;
    int b_tile_x = tid % BN;
    int b_tile_stride = thread_num / BN;

    float accum[TM][TN] = {0.0f};
    for (int k = 0; k < K; k += BK) {
        // 从global memory搬运到shared memory
        // 每个线程搬运 BM/a_tile_stride个数据，这些数据间隔为a_tile_stride
        for (int i = 0; i < BM; i += a_tile_stride) {
            As[a_tile_y + i][a_tile_x] = A[(a_tile_y + i) * K + a_tile_x];
        }
        for (int i = 0; i < BK; i += b_tile_stride) {
            Bs[b_tile_y + i][b_tile_x] = B[(b_tile_y + i) * N + b_tile_x];
        }
        __syncthreads();

        A += BK;
        B += BK * N;

        for (int row = 0; row < TM; row++) {
            for (int col = 0; col < TN; col++) {
                for (int i = 0; i < BK; i++) {
                    // tx,ty:(0,8,16,...,120)
                    // row,col:(0~7)
                    // 对于线程[ty，tx]，它计算C中TM*TN个数据
                    // 对于As、Bs, 每一个数据的计算需要As的一行与Bs的一列(A一行的部分和B一列的部分)
                    // 计算一个accum[row][col]要访问As、Bs各TM*TN次，可以搬运shared memory数据到register memory中提高访问速度
                    accum[row][col] += As[ty + row][i] * Bs[i][tx + col];
                }
            }
        }
        __syncthreads();
    }

    for (int row = 0; row < TM; row++) {
        for (int col = 0; col < TN; col++) {
            C[(ty + row) * N + (tx + col)] = accum[row][col];
        }
    }
}
void iSgemmThreadTiled_ref_2d(int M, int N, int K,
                        const float* __restrict__ A, int lda,
                        const float* __restrict__ B, int ldb,
                        float* C, int ldc)
{
    dim3 block(16, 16);
    dim3 grid(CEIL(N, 128), CEIL(M, 128));

    kSgemmThreadTiled_ref_2d<128, 128, 8, 8, 8><<<grid, block>>>(A, B, C, M, N, K);
}



// 基于iSgemmThreadTiled_ref做float4向量化优化版本，先不修改A-->As-->register memory的转置，既存在float4访问并不合并的问题
// BM BN BK TM TN都属于数据布局的变量
template<const int BM,
         const int BN,
         const int BK,
         const int TM,
         const int TN>
__global__ void kSgemmThreadTiled_float4_noTranspose(const float* __restrict__ A, const float* __restrict__ B,
                        float* C, int M, int N, int K) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // 因为此核函数的线程布局blockSize是一维的，所以需要手动计算 threadIdx.x,threadIdx.y
    int block_dim_x = BN / TN;  // block中一行的thread数量 (128/8=16)
    int block_dim_y = BM / TM;  // block中一列的thread数量 (128/8=16)
    int thread_num = block_dim_x * block_dim_y;  // block中thread总量 (16*16=256)  ?blockDim.x

    // (threadTile操作(线程操作更多数据)，加倍Mul, 后续时填充+(0~Mul-1))
    int tx = (threadIdx.x % block_dim_x) * TN;  // threadtile左上角x坐标 ((0~255) % 16) * 8 --> (0~15) * 8 --> (0,8,16,...,120)  
    int ty = (threadIdx.x / block_dim_x) * TM;  // threadtile左上角y坐标 ((0~255) / 16) * 8 --> (0~15) * 8 --> (0,8,16,...,120)  

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // BM*BK是Block中的float数据总数，thread_num是Block中的thread总数
    // BM*BK/4表示Block中的float4数据总数，则 BM*BK/4/thread_num表示一个thread搬运多少个float4，也即一个thread搬运float4需要多少轮
    const int ldg_a_num = BM * BK / 4 / thread_num; 
    const int ldg_b_num = BK * BN / 4 / thread_num; 

    // 从global memory搬运到shared memory
    // 手动计算线程在As中的x，y方向坐标(以float4为单位)
    int a_tile_y = threadIdx.x / (BK / 4);
    int a_tile_x = threadIdx.x % (BK / 4 ) * 4;
    // 目前为止，已经将线程映射到了 As 矩阵的上方一部分区域。如果线程不够多，它们只能覆盖矩阵的上半截。
    int a_tile_stride = BM / ldg_a_num; 
    // threadIdx.x,thread_num是线程布局变量，BK是数据布局变量。
    /* int a_tile_stride = thread_num / BK; 
    // thread_num是线程布局变量block thread总量，BK是数据布局变量As矩阵的一行有多少列。 */
    // 物理意义：a_tile_stride是block所有线程一次性总共能搬运多少行。 

    int b_tile_y = threadIdx.x / (BN / 4); 
    int b_tile_x = threadIdx.x % (BN / 4) * 4;
    int b_tile_stride = BK / ldg_b_num;
    // int b_tile_stride = thread_num / BN; // threadIdx.x,thread_num是线程布局变量，BN是数据布局变量

    float accum[TM][TN] = {0.0f};
    float a_frag[TM] = {0.0f};
    float b_frag[TN] = {0.0f};

    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];
    
    #pragma unroll
    for (int k = 0; k < K; k += BK) {
        // 从global memory搬运到shared memory
        #pragma unroll
        for (int i = 0; i < BM; i += a_tile_stride) { // 一共有BM行
            FLOAT4(As[a_tile_y + i][a_tile_x]) = reinterpret_cast<const float4*>(&A[(a_tile_y + i) * K + a_tile_x])[0];
        }
        #pragma unroll
        for (int i = 0; i < BK; i += b_tile_stride) {
            FLOAT4(Bs[b_tile_y + i][b_tile_x]) = reinterpret_cast<const float4*>(&B[(b_tile_y + i) * N + b_tile_x])[0];
        }
        __syncthreads();

        A += BK;
        B += BK * N;

        
        // 外积，效率好。访问shared_memory TM+TN次 
        #pragma unroll
        for (int k_inner = 0; k_inner < BK; k_inner++) {
            // 搬运shared memory数据到register memory中提高访问速度
            // As的一列的TM个数据 As[BM][BK]
            // 因为A-->As仍以[BM][BK]排列,所以一个线程取As的一列TM个float。
            // 又因为要取的数据不是行主序，所以不能如Bs-->b_frag一样使用FLOAT4,只能element-wise取
            #pragma unroll
            for (int y = 0; y < TM; y++) {
                a_frag[y] = As[ty + y][k_inner];
            }
            // Bs的一行的TN个数据 Bs[BK][BN]
            #pragma unroll
            for (int x = 0; x < TN; x += 4) {
                FLOAT4(b_frag[x]) = FLOAT4(Bs[k_inner][tx + x]);
            }
            #pragma unroll
            for (int y = 0; y < TM; y++) {
                #pragma unroll
                for (int x = 0; x < TN; x++) {
                    accum[y][x] += a_frag[y] * b_frag[x];
                }
            }    
        }

        /* // 内积，效率比较低。访问shared_memoryTM*TN*2次
        #pragma unroll
        for (int k_inner = 0; k_inner < BK; k_inner++) {
            #pragma unroll
            for (int y = 0; y < TM; y += 4) {
                #pragma unroll
                for (int x = 0; x < TN; x += 4) {
                    accum[y][x] += As[ty + y][k_inner] * Bs[k_inner][tx + x];
                }
            }
        } */

        __syncthreads();
    }

    #pragma unroll
    for (int y = 0; y < TM; y++) {
        #pragma unroll
        for (int x = 0; x < TN; x += 4) {
            float4 c_tmp = FLOAT4(C[(ty + y) * N + (tx + x)]);
            c_tmp.x = accum[y][x   ];
            c_tmp.y = accum[y][x + 1];
            c_tmp.z = accum[y][x + 2];
            c_tmp.w = accum[y][x + 3];
            FLOAT4(C[(ty + y) * N + (tx + x)]) = c_tmp;
        }
    }
}
void iSgemmThreadTiled_float4_noTranspose(int M, int N, int K,
                        const float* __restrict__ A, int lda,
                        const float* __restrict__ B, int ldb,
                        float* C, int ldc)
{
    dim3 block(256);
    dim3 grid(CEIL(N, 128), CEIL(M, 128));

    kSgemmThreadTiled_float4_noTranspose<128, 128, 8, 8, 8><<<grid, block>>>(A, B, C, M, N, K);
}



template<const int BM,
         const int BN,
         const int BK,
         const int TM,
         const int TN>
__global__ void kSgemmThreadTiled_float4(const float* __restrict__ A, const float* __restrict__ B,
                        float* C, int M, int N, int K) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // 计算block线程布局(x,y和thread_num)
    const int block_dim_x = BN / TN;
    const int block_dim_y = BM / TM;
    const int thread_num = block_dim_x * block_dim_y; // 一个线程负责计算block中TM*TN个元素

    // 当前线程对应thread tile的左上角元素在block中的索引
    int tx = (threadIdx.x % block_dim_x) * TN;
    int ty = (threadIdx.x / block_dim_x) * TM;

    // __shared__ float As[BM][BK];
    // __shared__ float Bs[BK][BN]; 
    // 不再使用二维的形式初始化As,因为后面会将As转置,便于更自由分配数据
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // BM*BK是Block中的float数据总数，thread_num是Block中的thread总数
    // BM*BK/4表示Block中的float4数据总数，则 BM*BK/4/thread_num表示一个thread要搬运ldg_a_num个float4，也即一个thread搬运float4需要ldg_a_num轮
    const int ldg_a_num = BM * BK / 4 / thread_num; 
    const int ldg_b_num = BK * BN / 4 / thread_num; 

    // float4前，A和As的对于float的二维索引
    // float4后，A和As的对于float4的二维索引
    int a_tile_y = threadIdx.x / (BK / 4); 
    int a_tile_x = threadIdx.x % (BK / 4) * 4; // 行主序，最内层索引
    int a_tile_stride = BM / ldg_a_num; // 物理意义：a_tile_stride是block所有线程一次性总共能搬运多少行。 也可以(int a_tile_stride = thread_num / BK;?) 

    // float4前，B和Bs的对于float的二维索引
    // float4后，B和Bs的对于float4的二维索引
    int b_tile_y = threadIdx.x / (BN / 4); 
    int b_tile_x = threadIdx.x % (BN / 4) * 4; // 行主序，最内层索引
    int b_tile_stride = BK / ldg_b_num; 

    float accum[TM][TN] = {0.0f}; // 每个线程负责TM*TN个元素，则需要申请TM*TN个寄存器保存累加值

    // 
    float ldg_a_reg[4 * ldg_a_num] = {0.0f}; // 每个线程搬运ldg_a_num轮，寄存器缓存ldg_a_num个float4元素，用于转置As矩阵

    float a_frag[TM] = {0.0f};
    float b_frag[TN] = {0.0f};

    // 移动到当前block
    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    #pragma unroll
    for (int k = 0; k < K; k += BK) {
        #pragma unroll
        for (int i = 0; i < BM; i += a_tile_stride) {
            int ldg_index = i / a_tile_stride * 4;  // i/a_tile_stride就是循环次数(1,2,...,ldg_a_num)，4Mul用于取数
            FLOAT4(ldg_a_reg[ldg_index]) = reinterpret_cast<const float4*>(&A[(a_tile_y + i) * K + a_tile_x])[0]; // A-->A_register float4为单位
            // As转置存，其中ldg_a_reg做中间缓存，目的是读取时可以按FLOAT4读取
            // 以行主序读A,以行主序写ldg_a_reg,再以列主序写As，实现转置
            // A_register-->As  float为单位
            As[(a_tile_x    ) * BM + a_tile_y + i] = ldg_a_reg[ldg_index    ]; // 这里是BM不是BN,因为已经转置了
            As[(a_tile_x + 1) * BM + a_tile_y + i] = ldg_a_reg[ldg_index + 1];
            As[(a_tile_x + 2) * BM + a_tile_y + i] = ldg_a_reg[ldg_index + 2];
            As[(a_tile_x + 3) * BM + a_tile_y + i] = ldg_a_reg[ldg_index + 3];
        }
        #pragma unroll
        for (int i = 0; i < BK; i += b_tile_stride) {
            FLOAT4(Bs[(b_tile_y + i) * BN + b_tile_x]) = reinterpret_cast<const float4*>(&B[(b_tile_y + i) * N + b_tile_x])[0]; // 不需要转置
        }
        __syncthreads();

        A += BK;
        B += BK * N;

        #pragma unroll
        for (int k_inner = 0; k_inner < BK; k_inner++) {
            #pragma unroll
            for (int m = 0; m < TM; m += 4) {
                FLOAT4(a_frag[m]) = FLOAT4(As[k_inner * BM + ty + m]);
            }
            #pragma unroll
            for (int n = 0; n < TN; n += 4) {
                FLOAT4(b_frag[n]) = FLOAT4(Bs[k_inner * BN + tx + n]);
            }
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    accum[m][n] += a_frag[m] * b_frag[n];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int m = 0; m < TM; m++) {
        #pragma unroll
        for (int n = 0; n < TN; n += 4) {
            float4 ctmp = FLOAT4(C[(ty + m) * N + tx + n]);
            ctmp.x = accum[m][n    ];
            ctmp.y = accum[m][n + 1];
            ctmp.z = accum[m][n + 2];
            ctmp.w = accum[m][n + 3];
            FLOAT4(C[(ty + m) * N + tx + n]) = ctmp;
        }
    }
}
void iSgemmThreadTiled_float4(int M, int N, int K,
                        const float* __restrict__ A, int lda,
                        const float* __restrict__ B, int ldb,
                        float* C, int ldc)
{
    dim3 block(256);
    dim3 grid(CEIL(N, 128), CEIL(M, 128));

    kSgemmThreadTiled_float4<128, 128, 8, 8, 8><<<grid, block>>>(A, B, C, M, N, K);
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


    // allocate device memory
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
    std::cout << GREEN << std::endl  << __FILE__ << ":" << __LINE__ << 
    " [device navie]: elapsed = " << total_time / repeat_times << " ms " << RESET << std::endl;
    memset(kernel_C, 0, sizeC * sizeof(float));
    CHECK(cudaMemcpy(kernel_C, dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost));
    verifyResult(hC, kernel_C, M * N, 1e-3);  

    CHECK(cudaMemset(dC, 0, sizeC * sizeof(float)));
    total_time = TIME_RECORD(repeat_times, ([&]{iSgemmBlockTiled(M, N, K, dA, lda, dB, ldb, dC, ldc);}));
    std::cout << GREEN << std::endl  << __FILE__ << ":" << __LINE__ << 
    " [device block tile]: elapsed = " << total_time / repeat_times << " ms " << RESET << std::endl;
    memset(kernel_C, 0, sizeC * sizeof(float));
    CHECK(cudaMemcpy(kernel_C, dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost));
    verifyResult(hC, kernel_C, M * N, 1e-3);  

    CHECK(cudaMemset(dC, 0, sizeC * sizeof(float)));
    total_time = TIME_RECORD(repeat_times, ([&]{iSgemmBlockTiled_float4(M, N, K, dA, lda, dB, ldb, dC, ldc);}));
    std::cout << GREEN << std::endl  << __FILE__ << ":" << __LINE__ <<
    " [device block tile float4]: elapsed = " << total_time / repeat_times << " ms " << RESET << std::endl;
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
    std::cout << GREEN << std::endl  << __FILE__ << ":" << __LINE__ <<
    " [device thread tile]: elapsed = " << total_time / repeat_times << " ms " << RESET << std::endl;
    memset(kernel_C, 0, sizeC * sizeof(float));
    CHECK(cudaMemcpy(kernel_C, dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost));
    verifyResult(hC, kernel_C, M * N, 1e-3);  

    CHECK(cudaMemset(dC, 0, sizeC * sizeof(float)));
    total_time = TIME_RECORD(repeat_times, ([&]{iSgemmThreadTiled_ref(M, N, K, dA, lda, dB, ldb, dC, ldc);}));
    std::cout << GREEN << std::endl  << __FILE__ << ":" << __LINE__ << 
    " [device thread tile reference]: elapsed = " << total_time / repeat_times << " ms " << RESET << std::endl;
    memset(kernel_C, 0, sizeC * sizeof(float));
    CHECK(cudaMemcpy(kernel_C, dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost));
    verifyResult(hC, kernel_C, M * N, 1e-3);  

    CHECK(cudaMemset(dC, 0, sizeC * sizeof(float)));
    total_time = TIME_RECORD(repeat_times, ([&]{iSgemmThreadTiled_ref_2d(M, N, K, dA, lda, dB, ldb, dC, ldc);}));
    std::cout << GREEN << std::endl << __FILE__ << ":" << __LINE__ <<
    " [device thread tile reference 2d block]: elapsed = " << total_time / repeat_times << " ms " << RESET << std::endl;
    memset(kernel_C, 0, sizeC * sizeof(float));
    CHECK(cudaMemcpy(kernel_C, dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost));
    verifyResult(hC, kernel_C, M * N, 1e-3);  

    CHECK(cudaMemset(dC, 0, sizeC * sizeof(float)));
    total_time = TIME_RECORD(repeat_times, ([&]{iSgemmThreadTiled_float4_noTranspose(M, N, K, dA, lda, dB, ldb, dC, ldc);}));
    std::cout << GREEN << std::endl  << __FILE__ << ":" << __LINE__ <<
    " [device thread tile float4 without transpose]: elapsed = " << total_time / repeat_times << " ms " << RESET << std::endl;
    memset(kernel_C, 0, sizeC * sizeof(float));
    CHECK(cudaMemcpy(kernel_C, dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost));
    verifyResult(hC, kernel_C, M * N, 1e-3);  

    CHECK(cudaMemset(dC, 0, sizeC * sizeof(float)));
    total_time = TIME_RECORD(repeat_times, ([&]{iSgemmThreadTiled_float4(M, N, K, dA, lda, dB, ldb, dC, ldc);}));
    std::cout << GREEN << std::endl  << __FILE__ << ":" << __LINE__ <<
    " [device thread tile float4]: elapsed = " << total_time / repeat_times << " ms " << RESET << std::endl;
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
