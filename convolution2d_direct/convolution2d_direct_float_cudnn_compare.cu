#include "../common.h"
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <cudnn.h>

#define CPU
#define CUDA
#define CUDNN
#define INPUT_CHANNELS 3
#define OUTPUT_CHANNELS 1
#define KERNEL_SIZE 7
#define BLOCK_SIZE 16
#define STRIDE 1
#define PADDING 0
#define NTILING 8 // >=4, stride==1
#define TILE_SHARED (BLOCK_SIZE - 1) * STRIDE + KERNEL_SIZE 
#define TILE_SHARED_N (NTILING * BLOCK_SIZE - 1) * STRIDE + KERNEL_SIZE 
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

    memset(output, 0, sizeof(float) * Cout * OH * OW);

    for (int oc = 0; oc < Cout; ++oc) {
        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
                float sum = 0.0f;
                for (int c = 0; c < Cin; ++c) {
                    for (int ky = 0; ky < KH; ++ky) {
                        for (int kx = 0; kx < KW; ++kx) {
                            int in_y = oh * stride + ky - pad; // 计算输入特征图的y轴索引
                            int in_x = ow * stride + kx - pad; // x轴索引

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
// 对于pad，其实可以在·cpu端预填充避免kernel中大量if判断。预留pad保证完整逻辑
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

    float sum = 0.0f;

    int in_start_y = out_y * stride - pad; // 计算输入特征图的y轴索引base
    int in_start_x = out_x * stride - pad; // x轴索引base

    # pragma unroll
    for (int c = 0; c < Cin; ++c) {
        # pragma unroll
        for (int ky = 0; ky < KH; ++ky) {
            int in_y = in_start_y + ky;
            # pragma unroll
            for (int kx = 0; kx < KW; ++kx) {
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
    const float* __restrict__ input,   // [Cin][H][W]
    /* const float* __restrict__ kernel, */  // [Cout][Cin][KH][KW]
    float* __restrict__ output,        // [Cout][OH][OW]
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

    kConv2dDirect_naive<<<grid, block>>>(
        input, /* kernel, */output,
        Cin, H, W,
        Cout, KH, KW,
        OH, OW,
        stride, pad
    );
}



// gpu N Tiling
template<const int SHARED_SIZE_W,
         const int SHARED_SIZE_H, 
         const int N_TILE,
         const int CIN>
__global__ void kConv2dDirect_1x8_Tiling(
    const float* __restrict__ input,   // [Cin][H][W]
    float* __restrict__ output,        // [Cout][OH][OW]
    int Cin, int H, int W,
    int Cout, int KH, int KW,
    int OH, int OW,
    int stride, int pad  // stride == 1
) 
{
    __shared__ float s_input[SHARED_SIZE_H][SHARED_SIZE_W];

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

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    float sum4 = 0.0f;
    float sum5 = 0.0f;
    float sum6 = 0.0f;
    float sum7 = 0.0f;
    
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
                    input[c * H * W + in_y * W + in_x] : 0.0f; // 填充0
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

                float s0 = s_input[shared_idx_y][shared_x0];
                float s1 = s_input[shared_idx_y][shared_x0 + blockDim.x];
                float s2 = s_input[shared_idx_y][shared_x0 + blockDim.x * 2];
                float s3 = s_input[shared_idx_y][shared_x0 + blockDim.x * 3];
                float s4 = s_input[shared_idx_y][shared_x0 + blockDim.x * 4];
                float s5 = s_input[shared_idx_y][shared_x0 + blockDim.x * 5];
                float s6 = s_input[shared_idx_y][shared_x0 + blockDim.x * 6];
                float s7 = s_input[shared_idx_y][shared_x0 + blockDim.x * 7];

                /* sum0 = __fmaf_rn(s0, f, sum0);
                sum1 = __fmaf_rn(s1, f, sum1);
                sum2 = __fmaf_rn(s2, f, sum2);
                sum3 = __fmaf_rn(s3, f, sum3);
                sum4 = __fmaf_rn(s4, f, sum4);
                sum5 = __fmaf_rn(s5, f, sum5);
                sum6 = __fmaf_rn(s6, f, sum6);
                sum7 = __fmaf_rn(s7, f, sum7); */ // 更多耗时？

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
    const float* __restrict__ input,   // [Cin][H][W]
    /* kernel, */  // [Cout][Cin][KH][KW]
    float* __restrict__ output,        // [Cout][OH][OW]
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

    kConv2dDirect_1x8_Tiling<TILE_SHARED_N, TILE_SHARED, NTILING, INPUT_CHANNELS><<<grid, block>>>(
        input, output,
        Cin, H, W,
        Cout, KH, KW,
        OH, OW,
        1, pad 
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
              << "  Mismatched elements = " << num_bad << " / " << size << "\n" << "\n";
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
    int KH = KERNEL_SIZE, KW = KERNEL_SIZE;
    int stride = STRIDE, pad = PADDING;

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
    float *d_input;
    float *d_output;
    float *d_kernel;

    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_input), INPUT_CHANNELS * H * W * sizeof(float)));
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_kernel), OUTPUT_CHANNELS * INPUT_CHANNELS * KH * KW * sizeof(float)));
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_output), OUTPUT_CHANNELS * OH * OW * sizeof(float)));

    // copy H -> D
    CHECK(cudaMemcpy(d_input, h_input.data(), INPUT_CHANNELS * H * W * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_kernel, h_kernel.data(), OUTPUT_CHANNELS * INPUT_CHANNELS * KH * KW * sizeof(float), cudaMemcpyHostToDevice));
    // 卷积核尺寸小，使用常量内存加快访存速度。using cudaMemcpyToSymbol
    cudaMemcpyToSymbol(d_kernel_const, h_kernel.data(), OUTPUT_CHANNELS * INPUT_CHANNELS * KH * KW * sizeof(float), 0, cudaMemcpyHostToDevice);

#ifdef CPU
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
#endif

#ifdef CUDA
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


    // gpu N Tiling
    CHECK(cudaMemset(d_output, 0, OUTPUT_CHANNELS * OH * OW * sizeof(float)));
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
    memset(h_output_ref.data(), 0, OUTPUT_CHANNELS * OH * OW * sizeof(float));
    CHECK(cudaMemcpy(h_output_ref.data(), d_output, OUTPUT_CHANNELS * OH * OW * sizeof(float), cudaMemcpyDeviceToHost));
    verifyResult(h_output.data(), h_output_ref.data(), OUTPUT_CHANNELS * OH * OW);
#endif

#ifdef CUDNN
    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_desc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        input_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        1, INPUT_CHANNELS, H, W));

    CHECK_CUDNN(cudnnSetFilter4dDescriptor(
        filter_desc,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        OUTPUT_CHANNELS, INPUT_CHANNELS, KH, KW));

    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
        conv_desc,
        pad, pad,
        stride, stride,
        1, 1,
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        output_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        1, OUTPUT_CHANNELS, OH, OW));

    cudnnConvolutionFwdAlgo_t algo;
    int returnedAlgoCount = 0;
    cudnnConvolutionFwdAlgoPerf_t algoPerf;
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(
        cudnn,
        input_desc,
        filter_desc,
        conv_desc,
        output_desc,
        1,
        &returnedAlgoCount,
        &algoPerf));
    algo = algoPerf.algo;

    size_t workspace_bytes = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn,
        input_desc,
        filter_desc,
        conv_desc,
        output_desc,
        algo,
        &workspace_bytes));

    void* d_workspace = nullptr;
    if (workspace_bytes > 0) CHECK(cudaMalloc(&d_workspace, workspace_bytes));

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // warmup
    for (int i = 0; i < 10; ++i) {
        CHECK_CUDNN(cudnnConvolutionForward(
            cudnn,
            &alpha,
            input_desc, d_input,
            filter_desc, d_kernel,
            conv_desc, algo,
            d_workspace, workspace_bytes,
            &beta,
            output_desc, d_output));
    }
    CHECK(cudaDeviceSynchronize());

    // timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 100; ++i) {
        CHECK_CUDNN(cudnnConvolutionForward(
            cudnn,
            &alpha,
            input_desc, d_input,
            filter_desc, d_kernel,
            conv_desc, algo,
            d_workspace, workspace_bytes,
            &beta,
            output_desc, d_output));
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Avg forward time: " << ms / 100.0f << " ms" << std::endl;

    cudaFree(d_workspace);

    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn);

#endif

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;

}


/*
* ref:
* http://www.few.vu.nl/~bwn200/papers/werkhoven-a4mmc2011.pdf
*/
