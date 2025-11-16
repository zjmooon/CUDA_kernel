#include "../common.h"
#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>


__global__ void kImageUint8Resize_nearest(unsigned char* __restrict__ src, unsigned char* __restrict__ dst, 
                    int srcH, int srcW, int dstH, int dstW, float scaleH, float scaleW)
{
    int gx = threadIdx.x + blockDim.x * blockIdx.x;
    int gy = threadIdx.y + blockDim.y * blockIdx.y;

    int srcX = round(static_cast<float>(gx) * scaleW);
    int srcY = round(static_cast<float>(gy) * scaleH);

    if (srcX < 0 || srcX > srcW || srcY < 0 || srcY > srcH ) return;

    int dstIdx = (gy * dstW + gx) * 3;
    int srcIdx = (srcY * srcW + srcX) * 3;

    dst[dstIdx    ] = src[srcIdx    ];
    dst[dstIdx + 1] = src[srcIdx + 1];
    dst[dstIdx + 2] = src[srcIdx + 2];
}
// 最邻近插值，是否保持原高宽比由布尔值isKeepHW决定
void iImageUint8Resize_nearest(unsigned char* __restrict__ src, unsigned char* __restrict__ dst, 
                  int srcH, int srcW, int dstH, int dstW, bool isKeepHW) 
{
    // 线程布局以resized 的图像数据布局为根据。为每一个dstIdx 映射 srcIdx
    dim3 blockSize(16, 16);
    dim3 gridSize(CEIL(dstW, blockSize.x), CEIL(dstH, blockSize.y));

    float scaleH = static_cast<float>(srcH / dstH); 
    float scaleW = static_cast<float>(srcW / dstW); 
    float scale = scaleW > scaleH ? scaleW : scaleH;
    
    if (isKeepHW) {
        scaleW = scale;
        scaleH = scale;
    }
    kImageUint8Resize_nearest<<<gridSize, blockSize>>>(src, dst, srcH, srcW, dstH, dstW, scaleH, scaleW);
}


/* 
* 最邻近插值，图像保持原高宽比，且居中显示，偏移部分用0 padding
*/
__global__ void kImageUint8Resize_nearest_center(unsigned char* __restrict__ src, unsigned char* __restrict__ dst, 
                    int srcH, int srcW, int dstH, int dstW, float scaleH, float scaleW)
{
    int gx = threadIdx.x + blockDim.x * blockIdx.x;
    int gy = threadIdx.y + blockDim.y * blockIdx.y;

    int srcX = round(static_cast<float>(gx) * scaleW);
    int srcY = round(static_cast<float>(gy) * scaleH);

    if (srcX < 0 || srcX > srcW || srcY < 0 || srcY > srcH ) return;

    // 计算像素在目标图上的x,y方向上的偏移量
    gy = gy + int(dstH / 2) - int(srcH / (scaleH * 2));
    gx = gx + int(dstW / 2) - int(srcW / (scaleW * 2));

    int dstIdx = (gy * dstW + gx) * 3;
    int srcIdx = (srcY * srcW + srcX) * 3;

    dst[dstIdx    ] = src[srcIdx    ];
    dst[dstIdx + 1] = src[srcIdx + 1];
    dst[dstIdx + 2] = src[srcIdx + 2];
}
void iImageUint8Resize_nearest_center(unsigned char* __restrict__ src, unsigned char* __restrict__ dst, 
                  int srcH, int srcW, int dstH, int dstW) 
{
    /*
    * 线程布局以resized 的图像数据布局为根据。为每一个dstIdx 映射 srcIdx
    * src_x = dst_x * scale_x
    * src_y = dst_y * scale_y
    */ 
    dim3 blockSize(16, 16);
    dim3 gridSize(CEIL(dstW, blockSize.x), CEIL(dstH, blockSize.y));

    float scaleH = static_cast<float>(srcH / dstH); 
    float scaleW = static_cast<float>(srcW / dstW); 
    float scale = scaleW > scaleH ? scaleW : scaleH;
    
    scaleW = scale;
    scaleH = scale;

    kImageUint8Resize_nearest_center<<<gridSize, blockSize>>>(src, dst, srcH, srcW, dstH, dstW, scaleH, scaleW);
}


/*
* elementwise类型的算子优化方向: 
* 在block tile的基础上使用uint4向量化优化。
* Using vectorized loads reduces the total number of instructions, reduces latency, and improves bandwidth utilization.
* 线程布局在block维度除以4。
* https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/
*/


// 双线性插值 bilinear interpolation
__global__ void kImageUint8Resize_bilinear_center(unsigned char* __restrict__ src, unsigned char* __restrict__ dst, 
                    int srcH, int srcW, int dstH, int dstW, float scaleH, float scaleW)
{

    // resized图的坐标(gx, gy)
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    int gx = blockIdx.x * blockDim.x + threadIdx.x;

    // bilinear interpolation -- 计算x,y映射到原图时最近的4个坐标
    int src_y1 = floor((gy + 0.5) * scaleH - 0.5);
    int src_x1 = floor((gx + 0.5) * scaleW - 0.5);
    int src_y2 = src_y1 + 1;
    int src_x2 = src_x1 + 1;

    // src越界检查
    if (src_x1 < 0 || src_x1 > srcW || src_y1 < 0 || src_y1 > srcH) return; 

    // bilinear interpolation -- 计算原图上的坐标(浮点类型)在0~1之间的值
    float th = ((gy + 0.5) * scaleH - 0.5) - src_y1;
    float tw = ((gx + 0.5) * scaleW - 0.5) - src_x1;

    // bilinear interpolation -- 计算最近的4个面积
    float a1_1 = (1.0 - tw) * (1.0 - th);  //右下
    float a1_2 = tw * (1.0 - th);          //左下
    float a2_1 = (1.0 - tw) * th;          //右上
    float a2_2 = tw * th;                  //左上

    // bilinear interpolation -- 计算4个坐标所对应的索引
    int srcIdx1_1 = (src_y1 * srcW + src_x1) * 3;  //左上
    int srcIdx1_2 = (src_y1 * srcW + src_x2) * 3;  //右上
    int srcIdx2_1 = (src_y2 * srcW + src_x1) * 3;  //左下
    int srcIdx2_2 = (src_y2 * srcW + src_x2) * 3;  //右下

    // bilinear interpolation -- 计算原图在目标图中的x, y方向上的偏移量
    gy = gy + int(dstH / 2) - int(srcH / (scaleH * 2));
    gx = gx + int(dstW / 2) - int(srcW / (scaleW * 2));

    // bilinear interpolation -- 计算resized之后的图的索引
    int dstIdx = (gy * dstW  + gx) * 3;

    dst[dstIdx + 0] = round(
                        a1_1 * src[srcIdx1_1 + 0] + 
                        a1_2 * src[srcIdx1_2 + 0] +
                        a2_1 * src[srcIdx2_1 + 0] +
                        a2_2 * src[srcIdx2_2 + 0]);

    dst[dstIdx + 1] = round(
                        a1_1 * src[srcIdx1_1 + 1] + 
                        a1_2 * src[srcIdx1_2 + 1] +
                        a2_1 * src[srcIdx2_1 + 1] +
                        a2_2 * src[srcIdx2_2 + 1]);

    dst[dstIdx + 2] = round(
                        a1_1 * src[srcIdx1_1 + 2] + 
                        a1_2 * src[srcIdx1_2 + 2] +
                        a2_1 * src[srcIdx2_1 + 2] +
                        a2_2 * src[srcIdx2_2 + 2]);

}
void iImageUint8Resize_bilinear_center(unsigned char* __restrict__ src, unsigned char* __restrict__ dst, 
                  int srcH, int srcW, int dstH, int dstW) 
{
    dim3 blockSize(16, 16);
    dim3 gridSize(CEIL(dstW, blockSize.x), CEIL(dstH, blockSize.y));

    float scaleH = static_cast<float>(srcH / dstH); 
    float scaleW = static_cast<float>(srcW / dstW); 
    float scale = scaleW > scaleH ? scaleW : scaleH;
    
    scaleW = scale;
    scaleH = scale;

    kImageUint8Resize_bilinear_center<<<gridSize, blockSize>>>(src, dst, srcH, srcW, dstH, dstW, scaleH, scaleW);
}



int main() 
{
    int repeat_times = 10;
    float total_time;
    double iStart, iElaps;
    int dst_h = 640, dst_w = 640;

    cv::Mat h_src, opencv_dst;
    unsigned char *h_dst; 

    h_src = cv::imread("moon_3840_2160.jpg");
    const int src_bytes = h_src.rows * h_src.cols * h_src.channels() * sizeof(unsigned char);
    const int dst_bytes = dst_h * dst_w * 3 * sizeof(unsigned char);
    unsigned char *d_src, *d_dst;

    h_dst = static_cast<unsigned char*>(malloc(dst_bytes));

    // allocate device memory
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_src), src_bytes));
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_dst), dst_bytes));
    CHECK(cudaMemcpy(d_src, h_src.data, src_bytes, cudaMemcpyHostToDevice));

    // cpu opencv resize
    iStart = seconds();
    cv::resize(h_src, opencv_dst, cv::Size(dst_w, dst_h), 0, 0, cv::INTER_NEAREST);
    iElaps = seconds() - iStart;
    std::cout << GREEN << "[cpu nearest]: elapsed = " << iElaps * 1000 << " ms, " << RESET << std::endl << std::endl;
    cv::imwrite("cpu_nearest.jpg", opencv_dst);

    iStart = seconds();
    cv::resize(h_src, opencv_dst, cv::Size(dst_w, dst_h), 0, 0, cv::INTER_LINEAR);
    iElaps = seconds() - iStart;
    std::cout << GREEN << "[cpu bilinear]: elapsed = " << iElaps * 1000 << " ms, " << RESET << std::endl << std::endl;
    cv::imwrite("cpu_bilinear.jpg", opencv_dst);
    
    // gpu kernel
    CHECK(cudaMemset(d_dst, 0, dst_bytes));
    total_time = TIME_RECORD(repeat_times, ([&]{iImageUint8Resize_nearest(d_src, d_dst, h_src.rows, h_src.cols, dst_h, dst_w, false);}));
    std::cout << GREEN << std::endl << "[gpu nearest]: elapsed = " << total_time / repeat_times << " ms " << RESET << std::endl;
    memset(h_dst, 0, dst_bytes);
    CHECK(cudaMemcpy(h_dst, d_dst, dst_bytes, cudaMemcpyDeviceToHost));
    imageIO::saveBgrUint8(h_dst, dst_w, dst_h, "gpu_nearest_noKeepHW.jpg");

    CHECK(cudaMemset(d_dst, 0, dst_bytes));
    total_time = TIME_RECORD(repeat_times, ([&]{iImageUint8Resize_nearest(d_src, d_dst, h_src.rows, h_src.cols, dst_h, dst_w, true);}));
    std::cout << GREEN << std::endl << "[gpu nearest keep height:width ]: elapsed = " << total_time / repeat_times << " ms " << RESET << std::endl;
    memset(h_dst, 0, dst_bytes);
    CHECK(cudaMemcpy(h_dst, d_dst, dst_bytes, cudaMemcpyDeviceToHost));
    imageIO::saveBgrUint8(h_dst, dst_w, dst_h, "gpu_nearest_keepHW.jpg"); 

    CHECK(cudaMemset(d_dst, 0, dst_bytes));
    total_time = TIME_RECORD(repeat_times, ([&]{iImageUint8Resize_nearest_center(d_src, d_dst, h_src.rows, h_src.cols, dst_h, dst_w);}));
    std::cout << GREEN << std::endl << "[gpu nearest keep_height:width center]: elapsed = " << total_time / repeat_times << " ms " << RESET << std::endl;
    memset(h_dst, 0, dst_bytes);
    CHECK(cudaMemcpy(h_dst, d_dst, dst_bytes, cudaMemcpyDeviceToHost));
    imageIO::saveBgrUint8(h_dst, dst_w, dst_h, "gpu_nearest_center.jpg"); 

    CHECK(cudaMemset(d_dst, 0, dst_bytes));
    total_time = TIME_RECORD(repeat_times, ([&]{iImageUint8Resize_bilinear_center(d_src, d_dst, h_src.rows, h_src.cols, dst_h, dst_w);}));
    std::cout << GREEN << std::endl << "[gpu bilinear keep_height:width center]: elapsed = " << total_time / repeat_times << " ms " << RESET << std::endl;
    memset(h_dst, 0, dst_bytes);
    CHECK(cudaMemcpy(h_dst, d_dst, dst_bytes, cudaMemcpyDeviceToHost));
    imageIO::saveBgrUint8(h_dst, dst_w, dst_h, "gpu_bilinear_center.jpg");
    
    
    // cleanup
    free(h_dst);
    CHECK(cudaFree(d_src));
    CHECK(cudaFree(d_dst));
    
    return 0;
}