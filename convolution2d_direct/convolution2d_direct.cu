#include "../common.h"
#include <cuda_runtime.h>
#include <iostream>
#include <random>

#define INPUT_CHANNELS 3
#define OUTPUT_CHANNELS 1
#define KERNEL_SIZE 3
#define BLOCK_SIZE 16
#define STRIDE 1
#define PADDING 0
#define NTILING 8 // >=4, stride==1
#define TILE_SHARED (BLOCK_SIZE - 1) * STRIDE + KERNEL_SIZE 
#define TILE_SHARED_N (NTILING * BLOCK_SIZE - 1) * STRIDE + KERNEL_SIZE 
__constant__ int d_kernel_const[OUTPUT_CHANNELS * INPUT_CHANNELS * KERNEL_SIZE * KERNEL_SIZE];

int select_conv_kernel(int Cin, int Kh, int stride)
{
    if (stride > 1) {
        return 1;
    }

    if (Kh <= 3 && Cin <= 8) {
        return 4;
    }

    if (Kh <= 5 && Cin <= 16) {
        return 2;
    }

    return 1;
}
// cpu 
void conv2d_direct_cpu(
    const int* input,   // [Cin][H][W]
    const int* kernel,  // [Cout][Cin][KH][KW]
    int* output,        // [Cout][OH][OW]
    int Cin, int H, int W,
    int Cout, int KH, int KW,
    int stride, int pad
) {
    int OH = (H + 2 * pad - KH) / stride + 1;
    int OW = (W + 2 * pad - KW) / stride + 1;

    std::memset(output, 0, sizeof(int) * Cout * OH * OW);

    for (int oc = 0; oc < Cout; ++oc) {
        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
                int sum = 0.0f;
                for (int c = 0; c < Cin; ++c) {
                    for (int ky = 0; ky < KH; ++ky) {
                        for (int kx = 0; kx < KW; ++kx) {
                            int in_y = oh * stride + ky - pad;  // 计算输入特征图的高度索引
                            int in_x = ow * stride + kx - pad;

                            if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                                int in_idx = c * H * W + in_y * W + in_x;
                                int k_idx = oc * Cin * KH * KW + c * KH * KW + ky * KW + kx;

                                sum += input[in_idx] * kernel[k_idx];
                            }
                        }
                    }
                }
                output[oc * OH * OW + oh * OW + ow] = sum;
            }
        }
    }
}



// gpu naive
__global__ void kConv2dDirect_naive(
    const int* __restrict__ input,   // [Cin][H][W]
    /* const int* __restrict__ kernel, */  // [Cout][Cin][KH][KW]
    int* __restrict__ output,        // [Cout][OH][OW]
    int Cin, int H, int W,
    int Cout, int KH, int KW,
    int OH, int OW,
    int stride, int pad
) 
{
    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int out_c = blockIdx.z;

    if (out_x >= OW || out_y >= OH) return;

    int sum = 0;

    int in_start_y = out_y * stride - pad;
    int in_start_x = out_x * stride - pad;

    # pragma unroll
    for (int c = 0; c < Cin; ++c) {
        # pragma unroll
        for (int ky = 0; ky < KH; ++ky) {
            # pragma unroll
            for (int kx = 0; kx < KW; ++kx) {
                int in_y = in_start_y + ky;
                int in_x = in_start_x + kx;

                if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                    int in_idx =
                        c * H * W + in_y * W + in_x;
                    int k_idx =
                        out_c * Cin * KH * KW +
                        c * KH * KW +
                        ky * KW + kx;

                    sum += input[in_idx] * d_kernel_const[k_idx];
                }
            }
        }
    }

    output[out_c * OH * OW + out_y * OW + out_x] = sum;
}
void iConv2dDirect_naive(
    const int* __restrict__ input,   // [Cin][H][W]
    /* const int* __restrict__ kernel, */  // [Cout][Cin][KH][KW]
    int* __restrict__ output,        // [Cout][OH][OW]
    int Cin, int H, int W,
    int Cout, int KH, int KW,
    int OH, int OW,
    int stride, int pad ) 
{
    dim3 block(16, 16);
    dim3 grid(
        CEIL(OW, block.x),
        CEIL(OH, block.y),
        Cout
    );

    kConv2dDirect_naive<<<grid, block>>>(
        input, /* kernel, */output,
        Cin, H, W,
        Cout, KH, KW,
        OH, OW,
        stride, pad
    );
}



// gpu shared memory optimized / block tile
// https://github.com/eunomia-bpf/basic-cuda-tutorial/blob/main/06-cnn-convolution.cu
template<const int SHARED_SIZE, const int CIN = INPUT_CHANNELS>
__global__ void kConv2dDirect_blocked(
    const int* __restrict__ input,   // [Cin][H][W]
    int* __restrict__ output,        // [Cout][OH][OW]
    int Cin, int H, int W,
    int Cout, int KH, int KW,
    int OH, int OW,
    int stride, int pad
) 
{
    __shared__ int s_input[SHARED_SIZE][SHARED_SIZE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int out_x = blockIdx.x * blockDim.x + tx;
    const int out_y = blockIdx.y * blockDim.y + ty;
    const int out_c = blockIdx.z;
    
    // if (out_x >= OW || out_y >= OH) return;
    
    const int in_start_y = blockIdx.y * blockDim.y * stride - pad;
    const int in_start_x = blockIdx.x * blockDim.x * stride - pad;

    int sum = 0;
    
    # pragma unroll
    for (int c = 0; c < Cin; c++) {
        // global to shared memory 
        // 合并访存以及避免Bank Conflict
        // 1_1: 直接二维循环加载到共享内存
        # pragma unroll
        for (int y = ty; y < SHARED_SIZE; y += blockDim.y) {
            # pragma unroll
            for (int x = tx; x < SHARED_SIZE; x += blockDim.x) {
                int in_x = in_start_x + x;
                int in_y = in_start_y + y;

                /* int value = 0; 
                if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                    value = input[c * H * W + in_y * W + in_x];
                }
                
                if (x < SHARED_SIZE && y < SHARED_SIZE) {
                    s_input[c * SHARED_SIZE * SHARED_SIZE + y * SHARED_SIZE + x] = value;
                } */

                s_input[y][x] = (in_x >=0 && in_x < W && in_y >=0 && in_y < H) ? 
                    input[c * H * W + in_y * W + in_x] : 0; // 边界填充0
            }
        }
        __syncthreads();

        /* if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && out_x == 0 && out_y == 0 && out_c == 0) {
            for (int y = 0; y < SHARED_SIZE; y++) {
                for (int x = 0; x < SHARED_SIZE; x++) {
                    printf("[%d][%d][%d]=%d ", c, y, x, s_input[c][y][x]);
                }
                printf("\n");
            }
        } */

        /* // 1_2: 2D to 1D 
        # pragma unroll
        for (int i = ty * blockDim.x + tx; i < SHARED_SIZE * SHARED_SIZE; i += blockDim.x * blockDim.y) {
            int x = i % SHARED_SIZE;
            int y = i / SHARED_SIZE;

            int in_x = in_start_x + x;
            int in_y = in_start_y + y;

            s_input[c][y][x] = (in_x >=0 && in_x < W && in_y >=0 && in_y < H) ? 
                input[c * H * W + in_y * W + in_x] : 0; // 边界填充0
        } */
        
        // compute
        # pragma unroll
        for (int ky = 0; ky < KH; ++ky) {
            int shared_y = ty * stride + ky;
            # pragma unroll
            for (int kx = 0; kx < KW; ++kx) {
                // shared memory 索引
                int shared_x = tx * stride + kx;
                if (shared_x >= SHARED_SIZE && shared_y >= SHARED_SIZE) return;

                /* int shared_idx = c * SHARED_SIZE * SHARED_SIZE + 
                                 shared_y * SHARED_SIZE + shared_x; */
                int k_idx = out_c * Cin * KH * KW +
                            c * KH * KW +
                            ky * KW + kx;

                sum += s_input[shared_y][shared_x] * d_kernel_const[k_idx];
            }
        }
        __syncthreads();
    }

    if (out_x >= OW || out_y >= OH) return;
    output[out_c * OH * OW + out_y * OW + out_x] = sum;
}
void iConv2dDirect_blocked(
    const int* __restrict__ input,   // [Cin][H][W]
    /* kernel, */  // [Cout][Cin][KH][KW]
    int* __restrict__ output,        // [Cout][OH][OW]
    int Cin, int H, int W,
    int Cout, int KH, int KW,
    int OH, int OW,
    int stride, int pad ) 
{
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(
        CEIL(OW, block.x),
        CEIL(OH, block.y),
        Cout
    );

    /* const int tileSize = block.x;
    const int TILE_SHARED = (tileSize - 1) * stride + KERNEL_SIZE ; // 共享内存的宽高: 16 + 3 - 1 =18
    const int sharedMemBytes = INPUT_CHANNELS * TILE_SHARED * TILE_SHARED * sizeof(int); */

    kConv2dDirect_blocked<TILE_SHARED><<<grid, block>>>(
        input, output,
        Cin, H, W,
        Cout, KH, KW,
        OH, OW,
        stride, pad
    );
} 



template<const int SHARED_SIZE_W,
         const int SHARED_SIZE_H, 
         const int N_TILE,
         const int CIN>
__global__ void kConv2dDirect_1x2_Tiling(
    const int* __restrict__ input,   // [Cin][H][W]
    int* __restrict__ output,        // [Cout][OH][OW]
    int Cin, int H, int W,
    int Cout, int KH, int KW,
    int OH, int OW,
    int stride, int pad
) 
{
    __shared__ int s_input[SHARED_SIZE_H][SHARED_SIZE_W];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int out_x_0 = blockIdx.x * blockDim.x * N_TILE + tx;
    const int out_x_1 = out_x_0 + blockDim.x;

    const int out_y = blockIdx.y * blockDim.y + ty;
    const int out_c = blockIdx.z;
    
    const int in_start_y = blockIdx.y * blockDim.y * stride - pad;
    const int in_start_x = blockIdx.x * blockDim.x * N_TILE * stride - pad;

    int sum0 = 0;
    int sum1 = 0;
    
    for (int c = 0; c < Cin; c++) {
        // global to shared memory 
        // 合并访存以及避免Bank Conflict
        # pragma unroll
        for (int y = ty; y < SHARED_SIZE_H; y += blockDim.y) {
            # pragma unroll
            for (int x = tx; x < SHARED_SIZE_W; x += blockDim.x) {
                
                int in_y = in_start_y + y;
                int in_x = in_start_x + x;
            
                s_input[y][x] = (in_x >=0 && in_x < W && in_y >=0 && in_y < H) ? 
                    input[c * H * W + in_y * W + in_x] : 0; // 填充0
            }
        }
        __syncthreads();

        /* if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && out_x == 0 && out_y == 0 && out_c == 0) {
            for (int y = 0; y < SHARED_SIZE_H; y++) {
                for (int x = 0; x < SHARED_SIZE_W; x++) {
                    printf("[%d][%d][%d]=%d ", c, y, x, s_input[c][y][x]);
                }
                printf("\n");
            }
        } */
        
        // convolution compute
        # pragma unroll
        for (int ky = 0; ky < KH; ++ky) {
            int shared_idx_y = ty * stride + ky;
            # pragma unroll
            for (int kx = 0; kx < KW; ++kx) {
                int f = d_kernel_const[out_c * Cin * KH * KW +
                            c * KH * KW +
                            ky * KW + kx];
                
                // shared memory 索引映射
                int shared_x0 = tx * stride + kx;
                int shared_x1 = shared_x0 + blockDim.x * stride;
                int s0 = s_input[shared_idx_y][shared_x0];
                int s1 = s_input[shared_idx_y][shared_x1];
                sum0 += s0 * f;
                sum1 += s1 * f;
            }
        }
        __syncthreads();
    }

    if (out_x_0 < OW) 
        output[out_c * OH * OW + out_y * OW + out_x_0] = sum0;
    if (out_x_1 < OW)
        output[out_c * OH * OW + out_y * OW + out_x_1] = sum1;
    
}
template<const int SHARED_SIZE_W,
         const int SHARED_SIZE_H, 
         const int N_TILE,
         const int CIN>
__global__ void kConv2dDirect_1x4_Tiling(
    const int* __restrict__ input,   // [Cin][H][W]
    int* __restrict__ output,        // [Cout][OH][OW]
    int Cin, int H, int W,
    int Cout, int KH, int KW,
    int OH, int OW,
    int stride, int pad  // stride == 1
) 
{
    __shared__ int s_input[SHARED_SIZE_H][SHARED_SIZE_W];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int out_x_0 = blockIdx.x * blockDim.x * N_TILE + tx;
    const int out_x_1 = out_x_0 + blockDim.x;
    const int out_x_2 = out_x_1 + blockDim.x;
    const int out_x_3 = out_x_2 + blockDim.x;

    const int out_y = blockIdx.y * blockDim.y + ty;
    const int out_c = blockIdx.z;

    // if (out_y >= OH || out_c >= Cout) return;  // 会与后续的__syncthreads() 构成死锁

    const int in_start_y = blockIdx.y * blockDim.y - pad;
    const int in_start_x = blockIdx.x * blockDim.x * N_TILE - pad;

    int sum0 = 0;
    int sum1 = 0;
    int sum2 = 0;
    int sum3 = 0;
    
    for (int c = 0; c < Cin; c++) {
        // global to shared memory 
        // 合并访存以及避免Bank Conflict
        # pragma unroll
        for (int y = ty; y < SHARED_SIZE_H; y += blockDim.y) {
            # pragma unroll
            for (int x = tx; x < SHARED_SIZE_W; x += blockDim.x) {
                
                int in_y = in_start_y + y;
                int in_x = in_start_x + x;
            
                s_input[y][x] = (in_x >=0 && in_x < W && in_y >=0 && in_y < H) ? 
                    input[c * H * W + in_y * W + in_x] : 0; // 填充0
            }
        }
        __syncthreads();
        
        // convolution compute
        # pragma unroll
        for (int ky = 0; ky < KH; ++ky) {
            int shared_idx_y = ty + ky;
            # pragma unroll
            for (int kx = 0; kx < KW; ++kx) {
                int f = d_kernel_const[out_c * Cin * KH * KW +
                            c * KH * KW +
                            ky * KW + kx];
                
                // shared memory 索引映射
                int shared_x0 = tx + kx;

                int s0 = s_input[shared_idx_y][shared_x0];
                int s1 = s_input[shared_idx_y][shared_x0 + blockDim.x];
                int s2 = s_input[shared_idx_y][shared_x0 + blockDim.x * 2];
                int s3 = s_input[shared_idx_y][shared_x0 + blockDim.x * 3];

                sum0 += s0 * f;
                sum1 += s1 * f;
                sum2 += s2 * f;
                sum3 += s3 * f;
            }
        }
        __syncthreads();
    }

    int base = out_c * OH * OW + out_y * OW;
    if (out_x_0 < OW) output[base + out_x_0] = sum0;
    if (out_x_1 < OW) output[base + out_x_1] = sum1;
    if (out_x_2 < OW) output[base + out_x_2] = sum2;
    if (out_x_3 < OW) output[base + out_x_3] = sum3;
    
}
template<const int SHARED_SIZE_W,
         const int SHARED_SIZE_H, 
         const int N_TILE,
         const int CIN>
__global__ void kConv2dDirect_1x8_Tiling(
    const int* __restrict__ input,   // [Cin][H][W]
    int* __restrict__ output,        // [Cout][OH][OW]
    int Cin, int H, int W,
    int Cout, int KH, int KW,
    int OH, int OW,
    int stride, int pad  // stride == 1
) 
{
    __shared__ int s_input[SHARED_SIZE_H][SHARED_SIZE_W];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int out_x_0 = blockIdx.x * blockDim.x * N_TILE + tx;
    const int out_x_1 = out_x_0 + blockDim.x;
    const int out_x_2 = out_x_1 + blockDim.x;
    const int out_x_3 = out_x_2 + blockDim.x;
    const int out_x_4 = out_x_3 + blockDim.x;
    const int out_x_5 = out_x_4 + blockDim.x;
    const int out_x_6 = out_x_5 + blockDim.x;
    const int out_x_7 = out_x_6 + blockDim.x;

    const int out_y = blockIdx.y * blockDim.y + ty;
    const int out_c = blockIdx.z;

    // if (out_y >= OH || out_c >= Cout) return;  // 会与后续的__syncthreads() 构成死锁

    const int in_start_y = blockIdx.y * blockDim.y - pad;
    const int in_start_x = blockIdx.x * blockDim.x * N_TILE - pad;

    int sum0 = 0;
    int sum1 = 0;
    int sum2 = 0;
    int sum3 = 0;
    int sum4 = 0;
    int sum5 = 0;
    int sum6 = 0;
    int sum7 = 0;
    

    for (int c = 0; c < Cin; c++) {
        // global to shared memory 
        // 合并访存以及避免Bank Conflict
        # pragma unroll
        for (int y = ty; y < SHARED_SIZE_H; y += blockDim.y) {
            # pragma unroll
            for (int x = tx; x < SHARED_SIZE_W; x += blockDim.x) {
                
                int in_y = in_start_y + y;
                int in_x = in_start_x + x;
            
                s_input[y][x] = (in_x >=0 && in_x < W && in_y >=0 && in_y < H) ? 
                    input[c * H * W + in_y * W + in_x] : 0; // 填充0
            }
        }
        __syncthreads();
        
        // convolution compute
        # pragma unroll
        for (int ky = 0; ky < KH; ++ky) {
            int shared_idx_y = ty + ky;
            # pragma unroll
            for (int kx = 0; kx < KW; ++kx) {
                int f = d_kernel_const[out_c * Cin * KH * KW +
                            c * KH * KW +
                            ky * KW + kx];
                
                // shared memory 索引映射
                int shared_x0 = tx + kx;

                int s0 = s_input[shared_idx_y][shared_x0];
                int s1 = s_input[shared_idx_y][shared_x0 + blockDim.x];
                int s2 = s_input[shared_idx_y][shared_x0 + blockDim.x * 2];
                int s3 = s_input[shared_idx_y][shared_x0 + blockDim.x * 3];
                int s4 = s_input[shared_idx_y][shared_x0 + blockDim.x * 4];
                int s5 = s_input[shared_idx_y][shared_x0 + blockDim.x * 5];
                int s6 = s_input[shared_idx_y][shared_x0 + blockDim.x * 6];
                int s7 = s_input[shared_idx_y][shared_x0 + blockDim.x * 7];

                sum0 += s0 * f;
                sum1 += s1 * f;
                sum2 += s2 * f;
                sum3 += s3 * f;
                sum4 += s4 * f;
                sum5 += s5 * f;
                sum6 += s6 * f;
                sum7 += s7 * f;
            }
        }
        __syncthreads();
    }

    int base = out_c * OH * OW + out_y * OW;
    if (out_x_0 < OW) output[base + out_x_0] = sum0;
    if (out_x_1 < OW) output[base + out_x_1] = sum1;
    if (out_x_2 < OW) output[base + out_x_2] = sum2;
    if (out_x_3 < OW) output[base + out_x_3] = sum3;
    if (out_x_4 < OW) output[base + out_x_4] = sum4;
    if (out_x_5 < OW) output[base + out_x_5] = sum5;
    if (out_x_6 < OW) output[base + out_x_6] = sum6;
    if (out_x_7 < OW) output[base + out_x_7] = sum7;
    
}
void iConv2dDirect_N_Tiling(
    const int* __restrict__ input,   // [Cin][H][W]
    /* kernel, */  // [Cout][Cin][KH][KW]
    int* __restrict__ output,        // [Cout][OH][OW]
    int Cin, int H, int W,
    int Cout, int KH, int KW,
    int OH, int OW,
    int stride, int pad) 
{
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(
        CEIL(OW, block.x * NTILING),
        CEIL(OH, block.y),
        Cout
    );

    switch (NTILING)
    {
    case 2:
        kConv2dDirect_1x2_Tiling<TILE_SHARED_N, TILE_SHARED, NTILING, INPUT_CHANNELS><<<grid, block>>>(
            input, output,
            Cin, H, W,
            Cout, KH, KW,
            OH, OW,
            stride, pad
        );
        break;
    case 4:
        kConv2dDirect_1x4_Tiling<TILE_SHARED_N, TILE_SHARED, NTILING, INPUT_CHANNELS><<<grid, block>>>(
            input, output,
            Cin, H, W,
            Cout, KH, KW,
            OH, OW,
            1, pad
        );
        break;
    case 8:
        kConv2dDirect_1x8_Tiling<TILE_SHARED_N, TILE_SHARED, NTILING, INPUT_CHANNELS><<<grid, block>>>(
            input, output,
            Cin, H, W,
            Cout, KH, KW,
            OH, OW,
            1, pad 
        );
        break;
    default:
        break;
    }



} 



void init_random_input(std::vector<int>& input, int low = 0, int high = 65535){
    static std::mt19937 gen(123); // 固定 seed，但序列不会重复
    std::uniform_int_distribution<int> dist(low, high);

    for (auto& v : input) v = dist(gen);
}


void init_random_kernel(std::vector<int>& kernel, int low = -10, int high = 10) {
    static std::mt19937 gen(123); // 固定 seed，但序列不会重复
    std::uniform_int_distribution<int> dist(low, high);

    for (auto& v : kernel) v = dist(gen);
}

void verifyResult(const int* host, const int* kernel, size_t size, double eps = 1e-3)
{
    int max_abs_err = 0;
    int sum_abs_err = 0;
    size_t num_bad = 0;

    for (size_t i = 0; i < size; ++i)
    {
        int diff = std::fabs(host[i] - kernel[i]);
        int abs_ref = std::fabs(host[i]);
        int rel_err = (abs_ref > 1e-6) ? diff / abs_ref : diff;

        if (rel_err > eps) {
            ++num_bad;
            // std::cout << i << ": " << host[i] << ", kernel " << kernel[i] << std::endl;
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


/*
* 在 GPU 上做卷积最常见的三类方法：
* 1. GEMM-based 卷积（im2col + GEMM）
* 把卷积变成矩阵乘法，再用 cuBLAS/优化矩阵乘法库加速。（内存占用高）
* 
* 2. FFT/Winograd 卷积
* 面向较大卷积核或特定尺寸优化（频域卷积/优化算法）。
* 
* 3. 直接卷积（Direct Convolution）
* 直接按照卷积定义逐元素计算，不作 im2col 变换。性能受益于合理的内存访问与优化策略。
*
* Input  : N x Cin x H x W (batch N默认为1)
* Kernel : Cout x Cin x KH x KW
* Output : N x Cout x OH x OW
*/
int main() {
    int repeat_times = 10;
    float total_time;
    double iStart, iElaps;
    
    int H = 1024*8, W = 1024*8;
    int KH = KERNEL_SIZE, KW = KERNEL_SIZE;
    int stride = STRIDE, pad = PADDING;

    // 计算卷积后输出尺寸
    int OH = (H + 2 * pad - KH) / stride + 1;
    int OW = (W + 2 * pad - KW) / stride + 1;

    std::vector<int> h_input(INPUT_CHANNELS * H * W);
    std::vector<int> h_kernel(OUTPUT_CHANNELS * INPUT_CHANNELS * KH * KW);
    std::vector<int> h_output(OUTPUT_CHANNELS * OH * OW);
    std::vector<int> h_output_ref(OUTPUT_CHANNELS * OH * OW);

    // 初始化输入和卷积核
    init_random_input(h_input);
    init_random_kernel(h_kernel);

    // device memory allocation
    int *d_input;
    int *d_output;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_input), INPUT_CHANNELS * H * W * sizeof(int)));
    // CHECK(cudaMalloc(reinterpret_cast<void **>(&d_kernel), OUTPUT_CHANNELS * INPUT_CHANNELS * KH * KW * sizeof(int)));
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_output), OUTPUT_CHANNELS * OH * OW * sizeof(int)));

    // copy H -> D
    CHECK(cudaMemcpy(d_input, h_input.data(), INPUT_CHANNELS * H * W * sizeof(int), cudaMemcpyHostToDevice));
    // CHECK(cudaMemcpy(d_kernel, h_kernel.data(), OUTPUT_CHANNELS * INPUT_CHANNELS * KH * KW * sizeof(int), cudaMemcpyHostToDevice));
    // 卷积核尺寸小，使用常量内存加快访存速度。using cudaMemcpyToSymbol
    cudaMemcpyToSymbol(d_kernel_const, h_kernel.data(), OUTPUT_CHANNELS * INPUT_CHANNELS * KH * KW * sizeof(int), 0, cudaMemcpyHostToDevice);

    iStart = seconds();
    conv2d_direct_cpu(
        h_input.data(),
        h_kernel.data(),
        h_output.data(),
        INPUT_CHANNELS, H, W,
        OUTPUT_CHANNELS, KH, KW,
        stride, pad
    ); 
    iElaps = seconds() - iStart;
    std::cout << GREEN << "[host]: elapsed = " << iElaps * 1000 << " ms " << RESET << std::endl << std::endl;

    // gpu naive (constant memory for kernel)
    CHECK(cudaMemset(d_output, 0, OUTPUT_CHANNELS * OH * OW * sizeof(int)));
    total_time = TIME_RECORD(repeat_times, ([&]{
        iConv2dDirect_naive(
            d_input,
            d_output,
            INPUT_CHANNELS, H, W,
            OUTPUT_CHANNELS, KH, KW,
            OH, OW,
            stride, pad
        );
    }));
    std::cout << GREEN << std::endl  << __FILE__ << ":" << __LINE__ << 
    " [device naive]: elapsed = " << total_time / repeat_times << " ms " << RESET << std::endl;
    memset(h_output_ref.data(), 0, OUTPUT_CHANNELS * OH * OW * sizeof(int));
    CHECK(cudaMemcpy(h_output_ref.data(), d_output, OUTPUT_CHANNELS * OH * OW * sizeof(int), cudaMemcpyDeviceToHost));
    verifyResult(h_output.data(), h_output_ref.data(), OUTPUT_CHANNELS * OH * OW);

    // gpu blocked
    CHECK(cudaMemset(d_output, 0, OUTPUT_CHANNELS * OH * OW * sizeof(int)));
    total_time = TIME_RECORD(repeat_times, ([&]{
        iConv2dDirect_blocked(
            d_input,
            d_output,
            INPUT_CHANNELS, H, W,
            OUTPUT_CHANNELS, KH, KW,
            OH, OW,
            stride, pad
        );
    }));
    std::cout << GREEN << std::endl  << __FILE__ << ":" << __LINE__ << 
    " [device blocked]: elapsed = " << total_time / repeat_times << " ms " << RESET << std::endl;
    memset(h_output_ref.data(), 0, OUTPUT_CHANNELS * OH * OW * sizeof(int));
    CHECK(cudaMemcpy(h_output_ref.data(), d_output, OUTPUT_CHANNELS * OH * OW * sizeof(int), cudaMemcpyDeviceToHost));
    verifyResult(h_output.data(), h_output_ref.data(), OUTPUT_CHANNELS * OH * OW);

    // gpu N Tiling
    CHECK(cudaMemset(d_output, 0, OUTPUT_CHANNELS * OH * OW * sizeof(int)));
    total_time = TIME_RECORD(repeat_times, ([&]{
        iConv2dDirect_N_Tiling(
            d_input,
            d_output,
            INPUT_CHANNELS, H, W,
            OUTPUT_CHANNELS, KH, KW,
            OH, OW,
            stride, pad
        );
    }));
    std::cout << GREEN << std::endl  << __FILE__ << ":" << __LINE__ << 
    " [device 1x" << NTILING << " Tiling]: elapsed = " << total_time / repeat_times << " ms " << RESET << std::endl;
    memset(h_output_ref.data(), 0, OUTPUT_CHANNELS * OH * OW * sizeof(int));
    CHECK(cudaMemcpy(h_output_ref.data(), d_output, OUTPUT_CHANNELS * OH * OW * sizeof(int), cudaMemcpyDeviceToHost));
    verifyResult(h_output.data(), h_output_ref.data(), OUTPUT_CHANNELS * OH * OW);
    return 0;
}


/*
* ref:
* http://www.few.vu.nl/~bwn200/papers/werkhoven-a4mmc2011.pdf
*/
