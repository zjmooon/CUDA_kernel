#include <iostream>
#include <vector>
#include <cassert>

#include <npp.h>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

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
    int stride, int pad,  // stride == 1
    float sigma_r // 值域标准差
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
    
    // 8个像素的累加器和权重归一化因子
    int sum0 = 0, sum1 = 0 , sum2 = 0, sum3 = 0, sum4 = 0, sum5 = 0, sum6 = 0, sum7 = 0;
    float w_sum0 = 0.0f, w_sum1 = 0.0f, w_sum2 = 0.0f, w_sum3 = 0.0f, 
          w_sum4 = 0.0f, w_sum5 = 0.0f, w_sum6 = 0.0f, w_sum7 = 0.0f;
    const float inv_rev_sigma2 = -1.0f / (2.0f * sigma_r * sigma_r);
    /* int sum[8] = {0};
    float w_sum[8] = {0.0f}; */
    
    for (int c = 0; c < Cin; c++) {
        // global to shared memory 
        // 合并访存以及避免Bank Conflict
        # pragma unroll
        for (int y = ty; y < SHARED_SIZE_H; y += blockDim.y) {
            int in_y = in_start_y + y;
            # pragma unroll
            for (int x = tx; x < SHARED_SIZE_W; x += blockDim.x) {
                int in_x = in_start_x + x;
            
                s_input[y][x] = (in_x >= 0 && in_x < W && in_y >= 0 && in_y < H) ? 
                    input[c * H * W + in_y * W + in_x] : 0; // 填充0
            }
        }
        __syncthreads();

        //  8个值域中心像素值
        int center_vals0, center_vals1, center_vals2, center_vals3, center_vals4, center_vals5, center_vals6, center_vals7;
        center_vals0 = s_input[ty + pad][tx + pad + 0 * blockDim.x];
        center_vals1 = s_input[ty + pad][tx + pad + 1 * blockDim.x];
        center_vals2 = s_input[ty + pad][tx + pad + 2 * blockDim.x];
        center_vals3 = s_input[ty + pad][tx + pad + 3 * blockDim.x];
        center_vals4 = s_input[ty + pad][tx + pad + 4 * blockDim.x];
        center_vals5 = s_input[ty + pad][tx + pad + 5 * blockDim.x];
        center_vals6 = s_input[ty + pad][tx + pad + 6 * blockDim.x];
        center_vals7 = s_input[ty + pad][tx + pad + 7 * blockDim.x];
        /* # pragma unroll
        for(int i = 0; i < 8; ++i) {
            center_vals[i] = s_input[ty + pad][tx + pad + i * blockDim.x];
        } */
        
        
        // convolution compute
        # pragma unroll
        for (int ky = 0; ky < KH; ++ky) {
            int shared_idx_y = ty + ky;
            # pragma unroll
            for (int kx = 0; kx < KW; ++kx) {
                // 空域权重(预计算查表)
                int s_weight = d_kernel_const[out_c * Cin * KH * KW +
                            c * KH * KW +
                            ky * KW + kx];
                // 总权重
                /* # pragma unroll
                for (int i = 0; i < 8; ++i) {
                    int shared_idx_x = tx + kx + i * blockDim.x;
                    float s = s_input[shared_idx_y][shared_idx_x];
                    
                    // 值域权重: exp(-|I(p)-I(q)|^2 / 2*sigma_r^2)
                    int diff = center_vals[i] - s;
                    float r_weight = expf(diff * diff * inv_rev_sigma2);
                    
                    float total_weight = s_weight * r_weight;
                    sum[i] += s * total_weight;
                    w_sum[i] += total_weight;
                } */

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
                
                // 值域权重: exp(-|I(p)-I(q)|^2 / 2*sigma_r^2)
                int diff0 = center_vals0 - s0;
                int diff1 = center_vals1 - s1;
                int diff2 = center_vals2 - s2;
                int diff3 = center_vals3 - s3;      
                int diff4 = center_vals4 - s4;
                int diff5 = center_vals5 - s5;
                int diff6 = center_vals6 - s6;
                int diff7 = center_vals7 - s7;
                
                // 总权重
                float r_weight, total_weight;

                r_weight = expf(diff0 * diff0 * inv_rev_sigma2);    
                total_weight = s_weight * r_weight;    
                w_sum0 += total_weight;  
                sum0 += s0 * total_weight;

                r_weight = expf(diff1 * diff1 * inv_rev_sigma2);    
                total_weight = s_weight * r_weight;    
                w_sum1 += total_weight;  
                sum1 += s1 * total_weight;

                r_weight = expf(diff2 * diff2 * inv_rev_sigma2);    
                total_weight = s_weight * r_weight;    
                w_sum2 += total_weight;  
                sum2 += s2 * total_weight;

                r_weight = expf(diff3 * diff3 * inv_rev_sigma2);    
                total_weight = s_weight * r_weight;    
                w_sum3 += total_weight;  
                sum3 += s3 * total_weight;

                r_weight = expf(diff4 * diff4 * inv_rev_sigma2);    
                total_weight = s_weight * r_weight;    
                w_sum4 += total_weight;  
                sum4 += s4 * total_weight;

                r_weight = expf(diff5 * diff5 * inv_rev_sigma2);    
                total_weight = s_weight * r_weight;    
                w_sum5 += total_weight;  
                sum5 += s5 * total_weight;

                r_weight = expf(diff6 * diff6 * inv_rev_sigma2);    
                total_weight = s_weight * r_weight;    
                w_sum6 += total_weight;  
                sum6 += s6 * total_weight;

                r_weight = expf(diff7 * diff7 * inv_rev_sigma2);    
                total_weight = s_weight * r_weight;    
                w_sum7 += total_weight;  
                sum7 += s7 * total_weight;

            }
        }
        __syncthreads();
    }

    int base = out_c * OH * OW + out_y * OW;
    if (out_x_0 < OW) output[base + out_x_0] = sum0 / (w_sum0 + 1e-6f);
    if (out_x_1 < OW) output[base + out_x_1] = sum1 / (w_sum1 + 1e-6f);
    if (out_x_2 < OW) output[base + out_x_2] = sum2 / (w_sum2 + 1e-6f);
    if (out_x_3 < OW) output[base + out_x_3] = sum3 / (w_sum3 + 1e-6f);
    if (out_x_4 < OW) output[base + out_x_4] = sum4 / (w_sum4 + 1e-6f);
    if (out_x_5 < OW) output[base + out_x_5] = sum5 / (w_sum5 + 1e-6f);
    if (out_x_6 < OW) output[base + out_x_6] = sum6 / (w_sum6 + 1e-6f);
    if (out_x_7 < OW) output[base + out_x_7] = sum7 / (w_sum7 + 1e-6f);
    
}

int main() {
    // 读取 CPU 图像
    int width, height, channels;
    unsigned char* input = stbi_load("../asset/moon_3840_2160.jpg", &width, &height, &channels, 3);
    if (!input) {
        std::cerr << "Failed to load image\n";
        return -1;
    }

    size_t imgSize = 3 * width * height * sizeof(unsigned char);

    // 分配 GPU 内存
    unsigned char* d_src;
    unsigned char* d_dst;
    cudaMalloc(&d_src, imgSize);
    cudaMalloc(&d_dst, imgSize);

    // 传输到 GPU
    cudaMemcpy(d_src, input, imgSize, cudaMemcpyHostToDevice);

    // 设置 NPP 参数
    NppiSize sizeROI = { width, height };
    int nRadius = 3;
    float valSigma = 25.0f;
    float posSigma = 5.0f;
    float valSquareSigma = valSigma * valSigma;
    float posSquareSigma = posSigma * posSigma;

    NppStatus status = nppiFilterBilateralGaussBorder_8u_C3R(
        d_src, 3 * width * sizeof(unsigned char),
        sizeROI, {0,0},
        d_dst, 3 * width * sizeof(unsigned char),
        sizeROI,
        nRadius,
        1,
        valSquareSigma,
        posSquareSigma,
        NPP_BORDER_REPLICATE
    );

    if (status != NPP_SUCCESS) {
        std::cerr << "NPP bilateral filter failed: " << status << "\n";
    }

    // 将结果转回 CPU
    std::vector<unsigned char> output(3 * height * width);
    cudaMemcpy(output.data(), d_dst, imgSize, cudaMemcpyDeviceToHost);

    // 保存
    stbi_write_png("../asset/moon_3840_2160_bilateraled_C3.png", width, height, 3, output.data(), width * 3);

    // 释放
    cudaFree(d_src);
    cudaFree(d_dst);
    stbi_image_free(input);

    std::cout << "Done!\n";
    return 0;
}
