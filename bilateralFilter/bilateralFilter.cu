#include "../common.h"
#include <iostream>
#include <vector>
#include <cassert>

#include <npp.h>
#include <cuda_runtime.h>

#define NPP_ENABLE true
#define CUDA_ENABLE true

constexpr float valSigma = 25.0f;
constexpr float posSigma = 5.0f;

#define RADIUS 3
#define PADDING 3 // 为了保持输入输出同尺寸，与 RADIUS 保持一致
#define BLOCK_SIZE 16
#define KERNEL_SIZE (2 * RADIUS + 1)
#define TILE_SHARED (BLOCK_SIZE - 1) + KERNEL_SIZE
#define TILE_SHARED_8 (8 * BLOCK_SIZE - 1) + KERNEL_SIZE
__constant__ float d_spatial_weights_const[KERNEL_SIZE * KERNEL_SIZE];

template<const int SHARED_SIZE_W,
         const int SHARED_SIZE_H, 
         const int N_TILE,
         const int CIN>
__global__ void kConv2dDirect_1x8_Tiling(
    const unsigned short* __restrict__ input,   // [Cin][H][W]
    unsigned short* __restrict__ output,        // [Cout][OH][OW]
    int Cin, int H, int W,
    int Cout, int OH, int OW, 
    int kernel_size, int padding, float sigma_r // range sigma (值域标准差)
) 
{
    __shared__ unsigned short s_input[SHARED_SIZE_H][SHARED_SIZE_W];
    float* d_spatial_weights_reg = d_spatial_weights_const;

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

    const int in_start_y = blockIdx.y * blockDim.y - padding;
    const int in_start_x = blockIdx.x * blockDim.x * N_TILE - padding;
    
    // 8个像素的累加器和权重归一化因子
    float sum0 = 0.0f, sum1 = 0.0f , sum2 = 0.0f, sum3 = 0.0f, sum4 = 0.0f, sum5 = 0.0f, sum6 = 0.0f, sum7 = 0.0f;
    float w_sum0 = 0.0f, w_sum1 = 0.0f, w_sum2 = 0.0f, w_sum3 = 0.0f, 
          w_sum4 = 0.0f, w_sum5 = 0.0f, w_sum6 = 0.0f, w_sum7 = 0.0f;
    const float inv_rev_sigma2 = -0.5f / (sigma_r * sigma_r);
    
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
        unsigned short center_vals0, center_vals1, center_vals2, center_vals3, center_vals4, center_vals5, center_vals6, center_vals7;
        center_vals0 = s_input[ty][tx + 0 * blockDim.x];
        center_vals1 = s_input[ty][tx + 1 * blockDim.x];
        center_vals2 = s_input[ty][tx + 2 * blockDim.x];
        center_vals3 = s_input[ty][tx + 3 * blockDim.x];
        center_vals4 = s_input[ty][tx + 4 * blockDim.x];
        center_vals5 = s_input[ty][tx + 5 * blockDim.x];
        center_vals6 = s_input[ty][tx + 6 * blockDim.x];
        center_vals7 = s_input[ty][tx + 7 * blockDim.x];
        
        // convolution compute
        # pragma unroll
        for (int ky = 0; ky < kernel_size; ++ky) {
            int shared_idx_y = ty + ky;
            # pragma unroll
            for (int kx = 0; kx < kernel_size; ++kx) {
                // 空域权重(预计算查表)
                float s_weight = d_spatial_weights_reg[out_c * Cin * kernel_size * kernel_size +
                            c * kernel_size * kernel_size +
                            ky * kernel_size + kx];

                // shared memory 索引映射
                int shared_x0 = tx + kx;

                unsigned short s0 = s_input[shared_idx_y][shared_x0];
                unsigned short s1 = s_input[shared_idx_y][shared_x0 + blockDim.x];
                unsigned short s2 = s_input[shared_idx_y][shared_x0 + blockDim.x * 2];
                unsigned short s3 = s_input[shared_idx_y][shared_x0 + blockDim.x * 3];
                unsigned short s4 = s_input[shared_idx_y][shared_x0 + blockDim.x * 4];
                unsigned short s5 = s_input[shared_idx_y][shared_x0 + blockDim.x * 5];
                unsigned short s6 = s_input[shared_idx_y][shared_x0 + blockDim.x * 6];
                unsigned short s7 = s_input[shared_idx_y][shared_x0 + blockDim.x * 7];
                
                // 值域权重: exp(-|I(p)-I(q)|^2 / 2*sigma_r^2)
                float diff0 = fabsf(center_vals0 - s0);
                float diff1 = fabsf(center_vals1 - s1);
                float diff2 = fabsf(center_vals2 - s2);
                float diff3 = fabsf(center_vals3 - s3);      
                float diff4 = fabsf(center_vals4 - s4);
                float diff5 = fabsf(center_vals5 - s5);
                float diff6 = fabsf(center_vals6 - s6);
                float diff7 = fabsf(center_vals7 - s7);
                
                // 总权重
                float r_weight, total_weight;

                r_weight = __expf(diff0 * diff0 * inv_rev_sigma2);    
                total_weight = __fmul_rn(s_weight, r_weight);    
                w_sum0 += total_weight;  
                sum0 += s0 * total_weight;

                r_weight = __expf(diff1 * diff1 * inv_rev_sigma2);    
                total_weight = __fmul_rn(s_weight, r_weight);    
                w_sum1 += total_weight;  
                sum1 += s1 * total_weight;

                r_weight = __expf(diff2 * diff2 * inv_rev_sigma2);    
                total_weight = __fmul_rn(s_weight, r_weight);    
                w_sum2 += total_weight;  
                sum2 += s2 * total_weight;

                r_weight = __expf(diff3 * diff3 * inv_rev_sigma2);    
                total_weight = __fmul_rn(s_weight, r_weight);    
                w_sum3 += total_weight;  
                sum3 += s3 * total_weight;

                r_weight = __expf(diff4 * diff4 * inv_rev_sigma2);    
                total_weight = __fmul_rn(s_weight, r_weight);    
                w_sum4 += total_weight;  
                sum4 += s4 * total_weight;

                r_weight = __expf(diff5 * diff5 * inv_rev_sigma2);    
                total_weight = __fmul_rn(s_weight, r_weight);    
                w_sum5 += total_weight;  
                sum5 += s5 * total_weight;

                r_weight = __expf(diff6 * diff6 * inv_rev_sigma2);    
                total_weight = __fmul_rn(s_weight, r_weight);    
                w_sum6 += total_weight;  
                sum6 += s6 * total_weight;

                r_weight = __expf(diff7 * diff7 * inv_rev_sigma2);    
                total_weight = __fmul_rn(s_weight, r_weight);    
                w_sum7 += total_weight;  
                sum7 += s7 * total_weight;

            }
        }
        __syncthreads();
    }

    int base = out_c * OH * OW + out_y * OW;
    if (out_x_0 < OW) output[base + out_x_0] = fmaxf(fmin(sum0 / (w_sum0 + 1e-6f), 65535.0f), 0.0f);
    if (out_x_1 < OW) output[base + out_x_1] = fmaxf(fmin(sum1 / (w_sum1 + 1e-6f), 65535.0f), 0.0f);
    if (out_x_2 < OW) output[base + out_x_2] = fmaxf(fmin(sum2 / (w_sum2 + 1e-6f), 65535.0f), 0.0f);
    if (out_x_3 < OW) output[base + out_x_3] = fmaxf(fmin(sum3 / (w_sum3 + 1e-6f), 65535.0f), 0.0f);
    if (out_x_4 < OW) output[base + out_x_4] = fmaxf(fmin(sum4 / (w_sum4 + 1e-6f), 65535.0f), 0.0f);
    if (out_x_5 < OW) output[base + out_x_5] = fmaxf(fmin(sum5 / (w_sum5 + 1e-6f), 65535.0f), 0.0f);
    if (out_x_6 < OW) output[base + out_x_6] = fmaxf(fmin(sum6 / (w_sum6 + 1e-6f), 65535.0f), 0.0f);
    if (out_x_7 < OW) output[base + out_x_7] = fmaxf(fmin(sum7 / (w_sum7 + 1e-6f), 65535.0f), 0.0f);    
    
}

void iConv2dDirect_8_Tiling(
    const unsigned short* __restrict__ input,   // [Cin][H][W]
    unsigned short* __restrict__ output,        // [Cout][OH][OW]
    int Cin, int H, int W,
    int Cout, int OH, int OW,
    int kernel_size, int padding, float valSigma) 
{
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(CEIL(OW, block.x * 8), CEIL(OH, block.y), Cout);

    kConv2dDirect_1x8_Tiling<TILE_SHARED_8, TILE_SHARED, 8, 1><<<grid, block>>>(
        input, output,
        Cin, H, W,
        Cout, OH, OW,
        kernel_size, padding, valSigma
    );
} 

void generateSpatialWeight(int nRadius, float nPosSquareSigma, float* h_spatial_weights) {
    int size = 2 * nRadius + 1;
    float normalization = 0.0f;
    
    float twoSigmaSquare = 2.0f * nPosSquareSigma; // NPP 传入的通常已经是 sigma^2
    
    for (int y = -nRadius; y <= nRadius; ++y) {
        for (int x = -nRadius; x <= nRadius; ++x) {
            float distanceSq = static_cast<float>(x * x + y * y);
            // 计算高斯分布
            float weight = std::exp(-distanceSq / twoSigmaSquare);
            normalization += weight;
            
            // 映射到数组索引 [0, 2*nRadius]
            h_spatial_weights[(y + nRadius) * size + (x + nRadius)] = weight;
        }
    }

    for (int y = -nRadius; y <= nRadius; ++y) {
        for (int x = -nRadius; x <= nRadius; ++x) {
            h_spatial_weights[(y + nRadius) * size + (x + nRadius)] /= normalization;
        }
    }
}

void callOpenCVBilateral(cv::Mat& input16u) {
    // 1. 转换为 32 位浮点数 (双边滤波在浮点下精度更高)
    cv::Mat input32f, output32f;
    input16u.convertTo(input32f, CV_32F);

    // 2. 参数设置
    int d = 2 * RADIUS + 1;       // 邻域直径 (NPP 的 nRadius = 3, 则 d = 7)
    double sigmaColor = valSigma;  // 对应你的 valSigma
    double sigmaSpace = posSigma;  // 对应你的 posSigma

    // 3. 调用函数
    cv::bilateralFilter(input32f, output32f, d, sigmaColor, sigmaSpace);

    // 4. 转回 16u 观察结果
    cv::Mat output16u;
    output32f.convertTo(output16u, CV_16U);
    
    cv::imwrite("asset/opencv_cpu_result.tif", output16u);
}

int main() {
    // load CPU 16u img
    int width, height;
    cv::Mat img = cv::imread("asset/moon_3840_2160_16u.tif", cv::IMREAD_UNCHANGED);
    if (img.empty()) { std::cerr << "Failed to load\n"; return -1; }
    width = img.cols;
    height = img.rows;

    size_t imgSize = width * height * sizeof(unsigned short);
    // 计算卷积后输出尺寸
    int OH = (height + 2 * PADDING - KERNEL_SIZE) + 1;
    int OW = (width + 2 * PADDING - KERNEL_SIZE) + 1;

    // alloc GPU memory
    unsigned short* d_src_16u;
    unsigned short* d_dst_16u;
    unsigned short* d_dst_16u_kernel;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_src_16u), height * width * sizeof(unsigned short)));
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_dst_16u), height * width * sizeof(unsigned short)));
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_dst_16u_kernel), OH * OW * sizeof(unsigned short)));
    // host -> device
    cudaMemcpy(d_src_16u, img.data, imgSize, cudaMemcpyHostToDevice);

    // opecv cpu
    callOpenCVBilateral(img);

    if (NPP_ENABLE) 
    {
        // set NPP params
        NppiSize sizeROI = { width, height };

        NppStatus status = nppiFilterBilateralGaussBorder_16u_C1R(
            d_src_16u, width * sizeof(unsigned short),
            sizeROI, {0,0},
            d_dst_16u, width * sizeof(unsigned short),
            sizeROI,
            RADIUS,
            1,
            valSigma * valSigma,
            posSigma * posSigma,
            NPP_BORDER_NONE
        );

        if (status != NPP_SUCCESS) {
            std::cerr << "NPP bilateral filter failed: " << status << "\n";
        }

        std::vector<unsigned short> output(height * width);
        cudaMemcpy(output.data(), d_dst_16u, imgSize, cudaMemcpyDeviceToHost);

        // build a 1-channel 16-bit OpenCV Mat and save as 16-bit TIFF
        cv::Mat outImg16u(height, width, CV_16UC1, output.data());
        cv::imwrite("asset/moon_3840_2160_npp_bilateraled_16u.tif", outImg16u);
    }

    if (CUDA_ENABLE) 
    {
        int repeat_times = 10;
        float total_time;

        float* h_spatial_weights = (float*)malloc(KERNEL_SIZE * KERNEL_SIZE * sizeof(float)); 
        generateSpatialWeight(RADIUS, posSigma * posSigma, h_spatial_weights);
        cudaMemcpyToSymbol(d_spatial_weights_const, h_spatial_weights, KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
        // gpu N Tiling
        CHECK(cudaMemset(d_dst_16u_kernel, 0, OH * OW * sizeof(unsigned short)));
        total_time = TIME_RECORD(repeat_times, ([&]{
            iConv2dDirect_8_Tiling(
                d_src_16u,
                d_dst_16u_kernel,
                1, height, width,
                1, OH, OW,
                KERNEL_SIZE, PADDING, valSigma
            );
        }));
        std::cout << GREEN << std::endl  << __FILE__ << ":" << __LINE__ << 
        " [device 1x8" << " Tiling]: elapsed = " << total_time / repeat_times << " ms " << RESET << std::endl;
        
        std::vector<unsigned short> output(OH * OW);
        CHECK(cudaMemcpy(output.data(), d_dst_16u_kernel, OH * OW * sizeof(unsigned short), cudaMemcpyDeviceToHost));
        cv::Mat outImg16u(height, width, CV_16UC1, output.data());
        cv::imwrite("asset/moon_" + std::to_string(OW) + "_" + std::to_string(OH) + "_kernel_bilateraled_16u.tif", outImg16u);
    }
 
    // free GPU memory
    cudaFree(d_src_16u);
    cudaFree(d_dst_16u);
    cudaFree(d_dst_16u_kernel);

    std::cout << "Done!\n";
    return 0;
}
