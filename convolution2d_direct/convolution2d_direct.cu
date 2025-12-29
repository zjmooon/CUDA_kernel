#include "../common.h"
#include <cuda_runtime.h>
#include <iostream>
#include <random>

#define INPUT_CHANNELS 3
#define OUTPUT_CHANNELS 2
#define KERNEL_SIZE 3
__constant__ float d_kernel_const[OUTPUT_CHANNELS * INPUT_CHANNELS * KERNEL_SIZE * KERNEL_SIZE];
 
// cpu 
void conv2d_direct_cpu(
    const float* input,   // [Cin][H][W]
    const float* kernel,  // [Cout][Cin][KH][KW]
    float* output,        // [Cout][OH][OW]
    int Cin, int H, int W,
    int Cout, int KH, int KW,
    int stride, int pad
) {
    int OH = (H + 2 * pad - KH) / stride + 1;
    int OW = (W + 2 * pad - KW) / stride + 1;

    std::memset(output, 0, sizeof(float) * Cout * OH * OW);

    for (int oc = 0; oc < Cout; ++oc) {
        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
                float sum = 0.0f;
                for (int c = 0; c < Cin; ++c) {
                    for (int kh = 0; kh < KH; ++kh) {
                        for (int kw = 0; kw < KW; ++kw) {
                            int ih = oh * stride + kh - pad;  // 计算输入特征图的高度索引
                            int iw = ow * stride + kw - pad;

                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                int in_idx = c * H * W + ih * W + iw;
                                int k_idx = oc * Cin * KH * KW + c * KH * KW + kh * KW + kw;

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
    const float* __restrict__ input,   // [Cin][H][W]
    /* const float* __restrict__ kernel, */  // [Cout][Cin][KH][KW]
    float* __restrict__ output,        // [Cout][OH][OW]
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

    float sum = 0.f;

    for (int c = 0; c < Cin; ++c) {
        for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
                int ih = out_y * stride + kh - pad;
                int iw = out_x * stride + kw - pad;

                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    int in_idx =
                        c * H * W + ih * W + iw;
                    int k_idx =
                        out_c * Cin * KH * KW +
                        c * KH * KW +
                        kh * KW + kw;

                    sum += input[in_idx] * d_kernel_const[k_idx];
                }
            }
        }
    }

    output[out_c * OH * OW + out_y * OW + out_x] = sum;
}
void iConv2dDirect_naive(
    const float* __restrict__ input,   // [Cin][H][W]
    /* const float* __restrict__ kernel, */  // [Cout][Cin][KH][KW]
    float* __restrict__ output,        // [Cout][OH][OW]
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
__global__ void kConv2dDirect_blocked(
    const float* __restrict__ input,   // [Cin][H][W]
    float* __restrict__ output,        // [Cout][OH][OW]
    int Cin, int H, int W,
    int Cout, int KH, int KW,
    int OH, int OW,
    int stride, int pad
) 
{
    extern __shared__ float s_input[];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int tile_size = blockDim.x;
    const int tileWithPad = tile_size + KERNEL_SIZE - 1;
    const int out_x = bx * tile_size + tx;
    const int out_y = by * tile_size + ty;
    const int out_c = blockIdx.z;
    
    if (out_x >= OW || out_y >= OH) return;
    
    // const int block_idx = tx + tile_size * ty ; // 二维block中的块内索引
    const int in_start_x = bx * tile_size * stride - pad;
    const int in_start_y = by * tile_size * stride - pad;
    
    // global to shared memory 
    // 合并访存以及避免Bank Conflict
    for (int c = 0; c < Cin; c++) {
        for (int y = ty; y < tileWithPad; y += tile_size) {
            for (int x = tx; x < tileWithPad; x += tile_size) {
                int in_x = in_start_x + x;
                int in_y = in_start_y + y;

                s_input[c * tileWithPad * tileWithPad + y * tileWithPad + x] = (in_x >=0 && in_x < W && in_y >=0 && in_y < H) ? 
                    input[c * H * W + in_y * W + in_x] : 0.f;
            }
        }
    }
    __syncthreads();

    
    // compute
    float sum = 0.f;
    for (int c = 0; c < Cin; ++c) {
        for (int ky = 0; ky < KH; ++ky) {
            for (int kx = 0; kx < KW; ++kx) {
                // shared memory 索引
                int shared_x = tx * stride + kx;
                int shared_y = ty * stride + ky;
                if (shared_x >= tileWithPad && shared_y >= tileWithPad) return;

                int shared_idx = c * tileWithPad * tileWithPad + 
                                 shared_y * tileWithPad + shared_x;
                int k_idx = out_c * Cin * KH * KW +
                            c * KH * KW +
                            ky * KW + kx;

                sum += s_input[shared_idx] * d_kernel_const[k_idx];
            }
        }
    }

    output[out_c * OH * OW + out_y * OW + out_x] = sum;
}
void iConv2dDirect_blocked(
    const float* __restrict__ input,   // [Cin][H][W]
    /* kernel, */  // [Cout][Cin][KH][KW]
    float* __restrict__ output,        // [Cout][OH][OW]
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

    const int tileSize = block.x;
    const int tileSizeWithPad = tileSize + KERNEL_SIZE- 1; 
    const int sharedMemBytes = INPUT_CHANNELS * tileSizeWithPad * tileSizeWithPad * sizeof(float);

    kConv2dDirect_blocked<<<grid, block, sharedMemBytes>>>(
        input, output,
        Cin, H, W,
        Cout, KH, KW,
        OH, OW,
        stride, pad
    );
}


void init_random(std::vector<float>& input, std::vector<float>& kernel, float low = 0.f, float high = 65535.f) {
    std::mt19937 gen(123); // 固定 seed，方便复现
    std::uniform_real_distribution<float> dist(low, high);

    for (auto& v : input)  v = dist(gen);
    for (auto& v : kernel) v = dist(gen);
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
    
    int H = 2160, W = 3840;
    int KH = 3, KW = 3;
    int stride = 1, pad = 0;

    // 计算卷积后输出尺寸
    int OH = (H + 2 * pad - KH) / stride + 1;
    int OW = (W + 2 * pad - KW) / stride + 1;

    std::vector<float> h_input(INPUT_CHANNELS * H * W);
    std::vector<float> h_kernel(OUTPUT_CHANNELS * INPUT_CHANNELS * KH * KW);
    std::vector<float> h_output(OUTPUT_CHANNELS * OH * OW);
    std::vector<float> h_output_ref(OUTPUT_CHANNELS * OH * OW);

    // 初始化输入和卷积核
    init_random(h_input, h_kernel);

    // device memory allocation
    float *d_input, *d_output;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_input), INPUT_CHANNELS * H * W * sizeof(float)));
    // CHECK(cudaMalloc(reinterpret_cast<void **>(&d_kernel), OUTPUT_CHANNELS * INPUT_CHANNELS * KH * KW * sizeof(float)));
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_output), OUTPUT_CHANNELS * OH * OW * sizeof(float)));

    // copy H -> D
    CHECK(cudaMemcpy(d_input, h_input.data(), INPUT_CHANNELS * H * W * sizeof(float), cudaMemcpyHostToDevice));
    // CHECK(cudaMemcpy(d_kernel, h_kernel.data(), OUTPUT_CHANNELS * INPUT_CHANNELS * KH * KW * sizeof(float), cudaMemcpyHostToDevice));
    // 卷积核尺寸小，使用常量内存加快访存速度。using cudaMemcpyToSymbol
    cudaMemcpyToSymbol(d_kernel_const, h_kernel.data(), OUTPUT_CHANNELS * INPUT_CHANNELS * KH * KW * sizeof(float), 0, cudaMemcpyHostToDevice);

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
    CHECK(cudaMemset(d_output, 0, OUTPUT_CHANNELS * OH * OW * sizeof(float)));
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
    memset(h_output_ref.data(), 0, OUTPUT_CHANNELS * OH * OW * sizeof(float));
    CHECK(cudaMemcpy(h_output_ref.data(), d_output, OUTPUT_CHANNELS * OH * OW * sizeof(float), cudaMemcpyDeviceToHost));
    verifyResult(h_output.data(), h_output_ref.data(), OUTPUT_CHANNELS * OH * OW);

    // gpu blocked
    CHECK(cudaMemset(d_output, 0, OUTPUT_CHANNELS * OH * OW * sizeof(float)));
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
    memset(h_output_ref.data(), 0, OUTPUT_CHANNELS * OH * OW * sizeof(float));
    CHECK(cudaMemcpy(h_output_ref.data(), d_output, OUTPUT_CHANNELS * OH * OW * sizeof(float), cudaMemcpyDeviceToHost));
    verifyResult(h_output.data(), h_output_ref.data(), OUTPUT_CHANNELS * OH * OW);


    return 0;
}
