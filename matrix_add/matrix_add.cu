#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include "../common.h"

constexpr int N = 3840 * 2160;

__global__ void elementwise_add_ushort(const ushort* A, const ushort* B, ushort* C, uint N) 
{    
    uint gid = threadIdx.x + blockIdx.x * blockDim.x;

    if (gid >= N) return;

    C[gid] = A[gid] + B[gid];
}

int main(int argc, char** argv) 
{
    ushort* h_A = reinterpret_cast<ushort *>(malloc(N * sizeof(ushort)));
    ushort* h_B = reinterpret_cast<ushort *>(malloc(N * sizeof(ushort)));
    ushort* h_C = reinterpret_cast<ushort *>(malloc(N * sizeof(ushort)));
    for (int i=0; i<N; ++i) {
        h_A[i] = 1;
        h_B[i] = 2;
    }

    ushort *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), N * sizeof(ushort)));
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), N * sizeof(ushort)));
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C), N * sizeof(ushort)));
    CHECK(cudaMemcpy(d_A, h_A, N * sizeof(ushort), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, N * sizeof(ushort), cudaMemcpyHostToDevice));

    dim3 block(32);
    dim3 grid(CEIL(N, block.x));
    
    // launch the kernel
    int kernelFun = 0;
    if (argc == 2) {
        kernelFun = atoi(argv[1]);
    }

    switch (kernelFun)
    {
    case 0:
        elementwise_add_ushort<<<grid, block>>>(d_A, d_B, d_C, N);
        CHECK(cudaMemcpy(h_C, d_C, N * sizeof(ushort), cudaMemcpyDeviceToHost));

        for (int i=0; i<N; ++i) {
            std::cout << i << ":" << h_C[i] << " ";
        }
        std::cout << std::endl;
        break;
    case 1:
        /* code */
        break;
    case 2:
        /* code */
        break;
    case 3:
        /* code */
        break;
    default:
        break;
    }

    return 0;
}