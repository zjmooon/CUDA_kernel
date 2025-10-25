#include "../common.h"
#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_SIZE 32

// host端做矩阵转置，与device端的结果进行比较
void transposeHost(const int *src, int *dst, const int nrows, const int ncols)
{
    for (int iy = 0; iy < nrows; ++iy)
    {
        for (int ix = 0; ix < ncols; ++ix)
        {
            dst[(ix * nrows) + iy] = src[((iy * ncols) + ix)];
        }
    }
}

void verifyResult(const int *hostMatrix, const int *deviceMatrix, const int nrows, const int ncols)
{
    for (int iy = 0; iy < nrows; ++iy)
    {
        for (int ix = 0; ix < ncols; ++ix)
        {
            if (hostMatrix[(iy * ncols) + ix] != deviceMatrix[(iy * ncols) + ix]) {
                std::cout << "result does not match in " << iy << "," << ix << std::endl;
                return;
            }

        }
    }
    std::cout << "succees! result match" << std::endl;
}

void printMatrix(const int *matrix, const int nrows, const int ncols)
{
    for (int iy = 0; iy < nrows; ++iy)
    {
        for (int ix = 0; ix < ncols; ++ix)
        {
            std::cout << matrix[(iy * ncols) + ix] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}



// device navie
__global__ void kTransposeNavie(const int *src, int *dst, const int M, const int N)
{
    const int gx = blockDim.x * blockIdx.x + threadIdx.x;
    const int gy = blockDim.y * blockIdx.y + threadIdx.y;

    if (gx < N && gy < M) {
        dst[gx * M + gy] = src[gy * N + gx];
    }
    /* 
    * load合并访存，store未合并访存
    * sector:物理层面概念，最小的load/store单位，32 bytes
    * load/store(SIMT): (1 << 13) * (1 << 12) (thread) / 32(thread / warp) = 1048576(instrument) --> 1048576(request)
    * load单个请求: 32 (thread) * 4 (byte / thead) = 128 byte --> sectors/Req = 128 / 32 = 4
    */
}
void iTransposeNavie(const int *src, int *dst, const int M, const int N, int* kernel_result) {
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));

    kTransposeNavie<<<gridSize, blockSize>>>(src, dst, M, N);

    // cudaMemcpy(kernel_result, dst, M * N * sizeof(int), cudaMemcpyDeviceToHost);
    // printMatrix(kernel_result, N, M);
}



// 合并写入(因为有缓存读load的机制，可以尽量提升性能，写store没有缓存机制，读load可以在空间和时间层面上进行缓存实现)
__global__ void kTransposeStoreCoalesce(const int *src, int *dst, const int M, const int N)
{
    const int gx = blockDim.x * blockIdx.x + threadIdx.x;
    const int gy = blockDim.y * blockIdx.y + threadIdx.y;

    if (gx < M && gy < N) {
        dst[gy * M + gx] = __ldg(&src[gx * N + gy]);
    }
}
void iTransposeStoreCoalesce(const int *src, int *dst, const int M, const int N, int* kernel_result) {
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));

    kTransposeStoreCoalesce<<<gridSize, blockSize>>>(src, dst, M, N);
}


// 使用shared memory 实现内存在读写上的合并
template<const int blockSize>
__global__ void kTransposeSharedCoalesce(const int *src, int *dst, const int M, const int N)
{
    const int bx = blockIdx.x * blockSize;
    const int by = blockIdx.y * blockSize;
    const int gx = bx + threadIdx.x;
    const int gy = by + threadIdx.y;
    __shared__ int s_data[blockSize][blockSize]; 

    if (gx < N && gy < M) {
        s_data[threadIdx.y][threadIdx.x] = src[gy * N + gx]; // 全局内存合并读取
    }
    __syncthreads();

    const int gx2 = by + threadIdx.x;
    const int gy2 = bx + threadIdx.y;

    if (gx2 < M && gy2 < N) {
        dst[gy2 * M + gx2] = s_data[threadIdx.x][threadIdx.y]; 
        // 全局内存合并写入 但共享内存Bank Conflict
    }

/* 
* 定义：相同线程束中(更严谨:相同memory transaction(load/store 128 bytes)，即可能小于32 线程)的不同线程访问相同Bank的不同地址。(只针对于共享内存)
* 所以向量化对于Bank Conflict有影响，一个线程不同的数据load/store会有不同的Bank Conflict效果
* 一共有32 bank，每个Bank32 bits 数据，或者64 bits
* 避免Bank Conflict: 1.padding(改变共享内存在Bank的布局) 2.swizzling(异或特性) 3.如果可以，调整线程访问模式为广播模式 
* https://forums.developer.nvidia.com/t/how-to-understand-the-bank-conflict-of-shared-mem/260900/2
*/
}
void iTransposeSharedCoalesce(const int *src, int *dst, const int M, const int N, int* kernel_result) {
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));

    kTransposeSharedCoalesce<BLOCK_SIZE><<<gridSize, blockSize>>>(src, dst, M, N);
}


// 使用shared memory 实现内存在读写上的合并 + padding避免Bank Conclict
template<const int blockSize>
__global__ void kTransposeSharedCoalescePadding(const int *src, int *dst, const int M, const int N)
{
    const int bx = blockIdx.x * blockSize;
    const int by = blockIdx.y * blockSize;
    const int gx = bx + threadIdx.x;
    const int gy = by + threadIdx.y;
    __shared__ int s_data[blockSize][blockSize + 1]; 
    /* 
    * 定义：相同线程束中(更严谨:相同memory transaction(load/store 128 bytes)，即可能小于32 线程)的不同线程访问相同Bank的不同地址。(只针对于共享内存)
    * 所以向量化对于Bank Conflict有影响，一个线程不同的数据load/store会有不同的Bank Conflict效果
    * 一共有32 bank，每个Bank32 bits 数据，或者64 bits
    * 避免Bank Conflict: 1.padding(改变共享内存在Bank的布局) 2.swizzling(异或特性) 3.如果可以，调整线程访问模式为广播模式 
    * https://forums.developer.nvidia.com/t/how-to-understand-the-bank-conflict-of-shared-mem/260900/2
    */

    if (gx < N && gy < M) {
        s_data[threadIdx.y][threadIdx.x] = src[gy * N + gx];    
    }
    __syncthreads();

    const int gx2 = by + threadIdx.x;
    const int gy2 = bx + threadIdx.y;

    if (gx2 < M && gy2 < N) {
        dst[gy2 * M + gx2] = s_data[threadIdx.x][threadIdx.y];
    }
}
void iTransposeSharedCoalescePadding(const int *src, int *dst, const int M, const int N, int* kernel_result) {
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));

    kTransposeSharedCoalescePadding<BLOCK_SIZE><<<gridSize, blockSize>>>(src, dst, M, N);
}


// 使用shared memory 实现内存在读写上的合并 + swizzling避免Bank Conclict
template<const int blockSize>
__global__ void kTransposeSharedCoalesceSwizzling(const int *src, int *dst, const int M, const int N)
{
    const int bx = blockIdx.x * blockSize;
    const int by = blockIdx.y * blockSize;
    const int gx = bx + threadIdx.x;
    const int gy = by + threadIdx.y;
    __shared__ int s_data[blockSize][blockSize]; 

    if (gx < N && gy < M) {
        s_data[threadIdx.y][threadIdx.x ^ threadIdx.y] = src[gy * N + gx];
    }
    __syncthreads();

    const int gx2 = by + threadIdx.x;
    const int gy2 = bx + threadIdx.y;

    if (gx2 < M && gy2 < N) {
        dst[gy2 * M + gx2] = s_data[threadIdx.x][threadIdx.x ^ threadIdx.y];
    }
}
void kTransposeSharedCoalesceSwizzling(const int *src, int *dst, const int M, const int N, int* kernel_result) {
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));

    kTransposeSharedCoalesceSwizzling<BLOCK_SIZE><<<gridSize, blockSize>>>(src, dst, M, N);
}



int main(int argc, char **argv)
{
    int repeat_times = 10;
    double iStart, iElaps;
    int N = 1 << 13; // N列
    int M = 1 << 12; // M行
    int total_size = M * N;
    float total_time;
    size_t bytes = total_size * sizeof(int);
    std::cout << "______________________With row: " << M << ", col: " << N << "______________________" << std::endl;

    // allocate host memory
    int *h_src = (int *)malloc(bytes);
    int *host_result = (int *)malloc(bytes);
    int *kernel_result = (int *)malloc(bytes);

    //  initialize host array
    for (int i = 0; i < total_size; i++)
    {
        h_src[i] = (int)(rand() & 0xFF);
    }
    // printMatrix(h_src, M, N);

    // allocate deveice memory
    int *d_src = nullptr, *d_dst = nullptr;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_src), bytes));
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_dst), bytes));

    // transpose at host  
    iStart = seconds();
    transposeHost(h_src, host_result, M, N);
    iElaps = seconds() - iStart;
    std::cout << RED << "[host]: elapsed = " << iElaps * 1000 << " ms, " << RESET << std::endl << std::endl;
    // printMatrix(host_result, N, M);

    // navie
    CHECK(cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice));
    total_time = TIME_RECORD(repeat_times, ([&]{iTransposeNavie(d_src, d_dst, M, N, kernel_result);}));
    std::cout << RED << std::endl << "[device navie]: elapsed = " << total_time / repeat_times << " ms " << RESET << std::endl;
    cudaMemcpy(kernel_result, d_dst, M * N * sizeof(int), cudaMemcpyDeviceToHost);
    verifyResult(host_result, kernel_result, N, M);

    // coalesced store
    CHECK(cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice));
    total_time = TIME_RECORD(repeat_times, ([&]{iTransposeStoreCoalesce(d_src, d_dst, M, N, kernel_result);}));
    std::cout << RED << std::endl << "[device coalesced store]: elapsed = " << total_time / repeat_times << " ms " << RESET << std::endl;
    cudaMemcpy(kernel_result, d_dst, M * N * sizeof(int), cudaMemcpyDeviceToHost);
    verifyResult(host_result, kernel_result, N, M); 

    // coalesced load&store in shared memory
    CHECK(cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice));
    total_time = TIME_RECORD(repeat_times, ([&]{iTransposeSharedCoalesce(d_src, d_dst, M, N, kernel_result);}));
    std::cout << RED << std::endl << "[device coalesced store&load]: elapsed = " << total_time / repeat_times << " ms " << RESET << std::endl;
    cudaMemcpy(kernel_result, d_dst, M * N * sizeof(int), cudaMemcpyDeviceToHost);
    verifyResult(host_result, kernel_result, N, M); 

    // coalesced load&store in shared memory + padding
    CHECK(cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice));
    total_time = TIME_RECORD(repeat_times, ([&]{iTransposeSharedCoalescePadding(d_src, d_dst, M, N, kernel_result);}));
    std::cout << RED << std::endl << "[device coalesced store&load padding]: elapsed = " << total_time / repeat_times << " ms " << RESET << std::endl;
    cudaMemcpy(kernel_result, d_dst, M * N * sizeof(int), cudaMemcpyDeviceToHost);
    verifyResult(host_result, kernel_result, N, M); 

    // coalesced load&store in shared memory + swizzling
    CHECK(cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice));
    total_time = TIME_RECORD(repeat_times, ([&]{kTransposeSharedCoalesceSwizzling(d_src, d_dst, M, N, kernel_result);}));
    std::cout << RED << std::endl << "[device coalesced store&load swizzling]: elapsed = " << total_time / repeat_times << " ms " << RESET << std::endl;
    cudaMemcpy(kernel_result, d_dst, M * N * sizeof(int), cudaMemcpyDeviceToHost);
    verifyResult(host_result, kernel_result, N, M); 

    // free host and device memory
    CHECK(cudaFree(d_src));
    CHECK(cudaFree(d_dst));
    free(h_src);
    free(host_result);
    free(kernel_result);

    return 1;
}