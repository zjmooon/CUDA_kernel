#include "../common.h"
#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_SIZE 32

// host端做矩阵转置，与device端的结果进行比较
void transposeHost(int *src, int *dst, const int nrows, const int ncols)
{
    for (int iy = 0; iy < nrows; ++iy)
    {
        for (int ix = 0; ix < ncols; ++ix)
        {
            dst[(ix * nrows) + iy] = src[((iy * ncols) + ix)];
        }
    }
}



// device navie
__global__ void kTransposeNavie(int *src, int *dst, const int M, const int N)
{
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    const int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col < N && row < M) {
        dst[col * M + row] = src[row * N + col];
    }
}
void iTransposeNavie(int *src, int *dst, const int M, const int N, int* kernel_res) {
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));

    kTransposeNavie<<<gridSize, blockSize>>>(src, dst, M, N);

    cudaMemcpy(kernel_res, dst, M * N, cudaMemcpyDeviceToHost);
}


// device, 合并写(因为有缓存读load的机制，可以尽量提升性能，写store没有缓存机制，读loa可以在空间和时间层面上进行缓存实现)
__global__ void kTransposeStoreCoalesce(int *src, int *dst, const int M, const int N)
{
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    const int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col < M && row < N) {
        dst[row * M + col] = __ldg(&dst[col * N + row]);
    }
}


int main(int argc, char **argv)
{
    int repeat_times = 10;
    double iStart, iElaps;
    int N = 1 << 20; // N列
    int M = 1 << 10; // M行
    int total_size = M * N;
    float total_time;
    size_t bytes = total_size * sizeof(int);
    std::cout << "______________________With row: " << M << ", col: " << N << "______________________" << std::endl;

    // allocate host memory
    int *h_src = (int *)malloc(bytes);
    int *cpu_src = (int *)malloc(bytes);
    int *kernel_src = (int *)malloc(bytes);

    //  initialize host array
    for (int i = 0; i < total_size; i++)
    {
        h_src[i] = (int)(rand() & 0xFF);
    }

    memcpy(cpu_src, h_src, bytes);

    // allocate deveice memory
    int *d_src = nullptr, *d_dst = nullptr;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_src), bytes));
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_dst), bytes));

    // transpose at host  
    iStart = seconds();
    transposeHost(h_src, cpu_src, M, N);
    iElaps = seconds() - iStart;
    std::cout << RED << "[host]: elapsed = " << iElaps * 1000 << " ms, " << RESET << std::endl << std::endl;

    // device navie
    CHECK(cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice));
    total_time = TIME_RECORD(repeat_times, ([&]{iTransposeNavie(d_src, d_dst, M, N, kernel_src);}));

    std::cout << RED << std::endl << "[device neighbored]: elapsed = " << total_time / repeat_times << " ms " << RESET << std::endl << std::endl;


    // free host and device memory
    CHECK(cudaFree(d_src));
    CHECK(cudaFree(d_dst));
    free(h_src);
    free(cpu_src);

    return 1;
}