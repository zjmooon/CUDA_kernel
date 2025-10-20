#include "../common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>


/*
 * This code implements the interleaved and neighbor-paired approaches to
 * parallel reduction in CUDA. For this example, the sum operation is used. A
 * variety of optimizations on parallel reduction aimed at reducing divergence
 * are also demonstrated, such as unrolling.
 */
// 归约问题：满足结合律和交换律的向量运算
// 常用的优化方式：循环展开，向量化，block内折半归约，warp内归约(天然同步)

// Recursive Implementation of Interleaved Pair Approach
int recursiveReduce(int *data, int const size)
{
    // terminate check
    if (size == 1) return data[0];

    // renew the stride
    int const stride = size / 2;

    // in-place reduction
    for (int i = 0; i < stride; i++)
    {
        data[i] += data[i + stride];
    }

    // call recursively
    return recursiveReduce(data, stride);
}



// Neighbored Pair Implementation with divergence
// 相邻配对归约，tid % (2 * stride) 工作线程并不连续(2,4,6,... --> 4,8,12,... )，导致线程束分化
__global__ void kReduceNeighbored(int *src, int *dst, unsigned int N)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    // g_idata + 偏移(blockIdx.x * blockDim.x：当前块在全局中的x方向上的索引)
    // blockIdx.x * blockDim.x：如果在一维网格一维块中
    // 就是表示当前 Block 的第一个线程在整个 Grid 中的全局线程编号起始值
    int *idata = src + blockIdx.x * blockDim.x; 

    // boundary check
    if (idx >= N) return;

    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if ((tid % (2 * stride)) == 0)
        {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    // if (tid == 0) dst[blockIdx.x] = idata[0];

    // 再做一次归约将每个块的值相加获得总的和
    if (tid == 0) atomicAdd(dst, idata[0]);
}
void iReduceNeighbored(int *src, int *dst, unsigned int n, int *sum) {
    dim3 blockSize(256, 1);
    dim3 gridSize(CEIL(n, blockSize.x), 1);
    
    cudaMemset(dst, 0, sizeof(int));

    kReduceNeighbored<<<gridSize, blockSize>>>(src, dst, n);
    CHECK(cudaDeviceSynchronize());

    *sum = 0;
    cudaMemcpy(sum, dst, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << *sum << ", ";
} 



/*
* Interleaved Pair Implementation with less divergence
* 交错归约，线程束分化程度降低(工作线程连续)，合并访存(全局内存的load/store连续)
*/
__global__ void kReduceInterleaved(int *src, int *dst, unsigned int N)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    // g_idata + 偏移(blockIdx.x * blockDim.x：当前块在全局中的x方向上的索引)
    // blockIdx.x * blockDim.x：如果在一维网格一维块中
    // 就是表示当前 Block 的第一个线程在整个 Grid 中的全局线程编号起始值
    int *idata = src + blockIdx.x * blockDim.x;   // 原始代码
    if (idx >= N) return;
    // idata 是指向当前 block数据 起始地址的指针。
    // 每个 block 的线程只对自己那一段连续的数组片段 idata[0 ~ blockDim.x-1] 进行归约。

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    // 每个 block 将自己的归约结果写到输出数组 dst 的对应位置。
    // 最终输出数组中会有 gridDim.x 个元素，每个是一个 block 的加和结果。
    // if (tid == 0) dst[blockIdx.x] = idata[0];

    // 每个block第一个线程，做一次归约将每个块的值相加获得总的和
    if (tid == 0) atomicAdd(dst, idata[0]);
}
void iReduceInterleaved(int *src, int *dst, unsigned int n, int *sum) {
    dim3 blockSize(256);
    dim3 gridSize(CEIL(n, blockSize.x));
    
    cudaMemset(dst, 0, sizeof(int));

    kReduceInterleaved<<<gridSize, blockSize>>>(src, dst, n);
    CHECK(cudaDeviceSynchronize());

    *sum = 0;
    cudaMemcpy(sum, dst, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << *sum << ", ";
} 



template<const int BLOCK_SIZE>
__global__ void kReduceInterleaved_shared(int *src, int *dst, unsigned int N)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int s_data[BLOCK_SIZE];    

    // boundary check & 搬运全局内存到共享内存（提升速度， 共享内存load/store性能 > 全局内存load/store性能）
    s_data[tid] = (idx < N) ? src[idx] : 0;
    __syncthreads();

    /* in-place reduction in shared memory
    * for循环从blockDim.x/2开始，消除Bank conflict
    * Bank conflict定义：只关注在shared memory, 相同warp的不同线程访问相同Bank的不同地址。一共32个Bank,每个Bank存储32bit或64bit
    * 更严谨的说法：同一memory transaction中的不同线程，既可能少于32个线程(https://forums.developer.nvidia.com/t/how-to-understand-the-bank-conflict-of-shared-mem/260900/2)
    */ 
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            s_data[tid] += s_data[tid + stride]; 
        }
        __syncthreads();
    }

    // 每个block第一个线程，做一次归约将每个块的值相加获得总的和
    if (tid == 0) atomicAdd(dst, s_data[0]);
}
void iReduceInterleaved_shared(int *src, int *dst, unsigned int n, int *sum) {
    dim3 blockSize(256, 1);
    dim3 gridSize(CEIL(n, blockSize.x), 1);
    
    cudaMemset(dst, 0, sizeof(int));

    kReduceInterleaved_shared<256><<<gridSize, blockSize>>>(src, dst, n);
    CHECK(cudaDeviceSynchronize());

    *sum = 0;
    cudaMemcpy(sum, dst, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << *sum << ", ";
} 



// warp reduce 
__global__ void kReduceWarp(const int *src, int *dst, int N) {
    __shared__ int s_data[32]; // 最多32， 因为一个block最多1024线程，最多1024/32=32warp

    int g_idx = threadIdx.x + blockDim.x * blockIdx.x;
    int warpId = threadIdx.x / warpSize; // 当前线程在其block中属于第几个warp
    int laneId = threadIdx.x % warpSize; // 当前线程在其warp中属于第几个thread

    int val = (g_idx < N) ? src[g_idx] : 0; // 将数据从全局内存搬运到寄存器内存, 三目运算符可有nvcc编译为selp指令（性能高于if）

    #pragma unroll
    for (int stride = warpSize / 2; stride > 0; stride >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, stride); // 将一个warp内的值归约到一个thread(laneId==0)中
    }

    if (laneId == 0) s_data[warpId] = val; // 将数据搬运到共享内存以便后续block内归约
    __syncthreads();

    if (warpId == 0) { // 在一个block内归约之前warp归约的结果
        int warpNum = blockDim.x / warpSize; 
        val = (laneId < warpNum) ? s_data[laneId] : 0; // 因为s_data可能没有全部用到，所以需要判断

        #pragma unroll
        for (int stride = warpSize / 2; stride > 0; stride >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, stride);
        }
        if (laneId == 0) atomicAdd(dst, val); // 一个block选择一个线程做atomicAdd，将全部block的值加到dst
    }
}
void iReduceWarp(int *src, int *dst, unsigned int n, int *sum) {
    dim3 blockSize(256);
    dim3 gridSize(CEIL(n, blockSize.x));
    
    cudaMemset(dst, 0, sizeof(int));

    kReduceWarp<<<gridSize, blockSize>>>(src, dst, n);
    CHECK(cudaDeviceSynchronize());

    *sum = 0;
    cudaMemcpy(sum, dst, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << *sum << ", ";
} 


/* 
* warp reduce vectorization4
*/
__global__ void kReduceWarpVec4(const int4 *src, int *dst, int N4) {
    __shared__ int s_data[32]; // 最多32， 因为一个block最多1024线程，最多1024/32=32warp

    int g_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (g_idx >= N4) return;
    int warpId = threadIdx.x / warpSize; // 当前线程在其block中属于第几个warp
    int laneId = threadIdx.x % warpSize; // 当前线程在其warp中属于第几个thread

    // int val = (g_idx < N4) ? src[g_idx] : 0; // 将数据从全局内存搬运到寄存器内存
    int val = 0;
    int4 tmp = src[g_idx];
    val += tmp.x;
    val += tmp.y;
    val += tmp.z;
    val += tmp.w;
    
    #pragma unroll
    for (int stride = warpSize / 2; stride > 0; stride >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, stride); // 将一个warp内的值归约到一个thread(laneId==0)中
    }

    if (laneId == 0) s_data[warpId] = val; // 将数据搬运到共享内存以便后续block内归约
    __syncthreads();

    if (warpId == 0) { // 在一个block内归约之前warp归约的结果
        int warpNum = blockDim.x / warpSize; 
        val = (laneId < warpNum) ? s_data[laneId] : 0; // 因为s_data可能没有全部用到，所以需要判断

        #pragma unroll
        for (int stride = warpSize / 2; stride > 0; stride >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, stride);
        }
        if (laneId == 0) atomicAdd(dst, val); // 一个block选择一个线程做atomicAdd，将全部block的值加到dst
    }
}
__global__ void kReduceWarpVec4_v2(const int4 *src, int *dst, int N4) {
    __shared__ int s_data[32];  // 可选：用动态 shared mem 或者固定大小（若确定）

    int tid = threadIdx.x;
    int g_idx = blockIdx.x * blockDim.x + tid;
    if (g_idx >= N4) {
        // 不参与：设 val = 0，确保安全
        // 注意：即使不参与读取 src，也要给一个默认 val = 0
        // 继续后续流程，使得 shuffle 和写 shared 不出错
        // 进入下面阶段
    }

    // 每线程读取一个 int4（如果在范围内）
    int4 tmp = make_int4(0, 0, 0, 0);
    if (g_idx < N4) {
        tmp = src[g_idx];
    }
    int val = tmp.x + tmp.y + tmp.z + tmp.w;

    // Warp 内归约：shuffle down
    unsigned mask = __ballot_sync(0xffffffff, g_idx < N4);  // 活跃线程 mask
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        int v2 = __shfl_down_sync(mask, val, offset);
        val += v2;
    }

    int laneId = tid & (warpSize - 1);
    int warpId = tid / warpSize;

    if (laneId == 0) {
        s_data[warpId] = val;
    }
    __syncthreads();

    // 在 block 内把每个 warp 的部分和归约
    int warpNum = (blockDim.x + warpSize - 1) / warpSize;  // 向上取整 warp 数量
    if (warpId == 0) {
        // 只有第一个 warp 执行此部分
        int partial = (laneId < warpNum) ? s_data[laneId] : 0;
        // 再次 warp 归约（仅在第一个 warp 内做）
        unsigned mask2 = __ballot_sync(0xffffffff, laneId < warpNum);
        #pragma unroll
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            int v2 = __shfl_down_sync(mask2, partial, offset);
            partial += v2;
        }
        if (laneId == 0) {
            atomicAdd(dst, partial);
        }
    }
}
void iReduceWarpVec4(int *src, int *dst, unsigned int n, int *sum) {
    dim3 blockSize(512);
    dim3 gridSize(CEIL(CEIL(n, 4), blockSize.x));  // 在grid维度取 1/4

    int4* packSrc = reinterpret_cast<int4*>(src); 
    cudaMemset(dst, 0, sizeof(int));
    
    kReduceWarpVec4<<<gridSize, blockSize>>>(packSrc, dst, CEIL(n, 4));
    CHECK(cudaDeviceSynchronize());

    *sum = 0;
    cudaMemcpy(sum, dst, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << *sum << ", ";
} 
void iReduceWarpVec4_v2(int *src, int *dst, unsigned int n, int *sum) {
    dim3 blockSize(512);
    dim3 gridSize(CEIL(CEIL(n, 4), blockSize.x));  // 在grid维度取 1/4

    int4* packSrc = reinterpret_cast<int4*>(src); 
    cudaMemset(dst, 0, sizeof(int));
    
    kReduceWarpVec4_v2<<<gridSize, blockSize>>>(packSrc, dst, CEIL(n, 4));
    CHECK(cudaDeviceSynchronize());

    *sum = 0;
    cudaMemcpy(sum, dst, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << *sum << ", ";
} 



template<const int BLOCK_SIZE>
__global__ void kReduceInterleavedUnrolling4(int *src, int *dst, unsigned int N)
{
    __shared__ int s_data[BLOCK_SIZE];
    unsigned int tid = threadIdx.x;

    /*
        循环展开通过减少分支出现的频率和循环维护指令，产生更高的指令和内存带宽用以帮助隐藏指令或内存延迟。
        idx 的计算方式是每个 Block 内的线程处理连续4倍的数据（通过 blockDim.x * 4），这是 unrolling 技术的一部分。
        若线程块blockDim.x=128， 一个线程块要处理的数据是 128*4，且跨度为blockDim.x。
    */
    unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

    // unrolling 4
    // 1个数据块处理原来4个数据块的数据(以blockDim.x为间隔，而非连续即非threadIdx.x)
    // 通过 idx + blockDim.x * 4 进行元素配对，减少了原始归约过程中的数据访问次数，使得每个线程处理4个元素。
    // 这有助于减少归约的总轮数，提高性能。
    // 因为归约的第一个阶段通过对邻近的两个元素进行加法操作，减少了元素总数，这样后续的归约迭代次数也会少一些。
    
    /* int sum = 0;
    if (idx + blockDim.x * 3 < N){
        sum = src[idx] + src[idx + blockDim.x] + src[idx + blockDim.x * 2] + src[idx + blockDim.x * 3];
    }  
    s_data[tid] = sum;
    __syncthreads(); */

    if (idx + blockDim.x * 3 < N){
        s_data[tid] = src[idx] + src[idx + blockDim.x] + src[idx + blockDim.x * 2] + src[idx + blockDim.x * 3];
    } 
    else {
        s_data[tid] = 0;
    } 
    __syncthreads();

    // 块内间隔归约
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride){
            s_data[tid] += s_data[tid +stride]; 
            // idata[tid] += idata[tid + stride];
            // idata[tid] == src[blockIdx.x * blockDim.x * 2 + threadIdx.x]
        }
        __syncthreads();
    }
    // stride 每轮减半，最后将块内所有值归约（加）到tid==0线程上。
    // idata[tid] += idata[tid + stride]会将一对数据进行加法并存回 idata。
    
    if (tid == 0) atomicAdd(dst, s_data[0]);
}
void iReduceInterleavedUnrolling4(int *src, int *dst, unsigned int n, int *sum) {
    dim3 blockSize(256);
    dim3 gridSize(CEIL(CEIL(n, blockSize.x), 4)); // 在grid维度取 1/4
    
    cudaMemset(dst, 0, sizeof(int));

    kReduceInterleavedUnrolling4<256><<<(gridSize), blockSize>>>(src, dst, n);
    CHECK(cudaDeviceSynchronize());

    *sum = 0;
    cudaMemcpy(sum, dst, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << *sum << ", ";
} 



__global__ void kReduceInterleavedUnrolling8(int *src, int *dst, unsigned int N)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;
    // idx是当前线程要操作的第一个数据的全局索引，以blockDim.x为步长
    int *idata = src + blockIdx.x * blockDim.x * 8;
    // idata是局部地址，是g_idata的某些偏移。blockIdx.x * blockDim.x * 8是线程层面转成数据层面（*8：每个线程处理8个数据）

    if (idx + blockDim.x * 7 < N) {
        // 全局指针g_idata 配全局索引idx
        src[idx] = src[idx] + src[idx + blockDim.x * 1] + src[idx + blockDim.x * 2] + src[idx + blockDim.x * 3] + 
                   src[idx + blockDim.x * 4] + src[idx + blockDim.x * 5] + src[idx + blockDim.x * 6] + src[idx + blockDim.x * 7]; 
    }
    __syncthreads(); // 需要放置栅栏

    for (int stride =  blockDim.x / 2; stride  > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) atomicAdd(dst, idata[0]);
}
__global__ void kReduceInterleavedUnrolling8_origin(int *src, int *dst, unsigned int N)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = src + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < N)
    {
        int a1 = src[idx];
        int a2 = src[idx + blockDim.x];
        int a3 = src[idx + 2 * blockDim.x];
        int a4 = src[idx + 3 * blockDim.x];
        int b1 = src[idx + 4 * blockDim.x];
        int b2 = src[idx + 5 * blockDim.x];
        int b3 = src[idx + 6 * blockDim.x];
        int b4 = src[idx + 7 * blockDim.x];
        src[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    if (tid == 0) atomicAdd(dst, idata[0]);
}
void iReduceInterleavedUnrolling8(int *src, int *dst, unsigned int n, int *sum) {
    dim3 blockSize(256);
    dim3 gridSize(CEIL(CEIL(n, blockSize.x), 8)); // 在grid维度取 1/8
    
    cudaMemset(dst, 0, sizeof(int));

    kReduceInterleavedUnrolling8<<<gridSize, blockSize>>>(src, dst, n);
    CHECK(cudaDeviceSynchronize());

    *sum = 0;
    cudaMemcpy(sum, dst, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << *sum << ", ";
} 

__global__ void reduceUnrolling8_unrollPragma(int *g_idata, int *g_odata, unsigned int n)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // 当前 block 的数据段起始地址
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    // Unroll 8 次读取
    if (idx + 7 * blockDim.x < n){
        int sum = 0;
        #pragma unroll 8
        for (int i = 0; i < 8; ++i)
        {
            sum += g_idata[idx + i * blockDim.x];
        }

        g_idata[idx] = sum;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1){
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0){
        g_odata[blockIdx.x] = idata[0];
    }
}

__global__ void kReduceUnrollWarps8(int *src, int *dst, unsigned int N)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = src + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < N)
    {
        src[idx] = src[idx] + src[idx + blockDim.x * 1] + src[idx + blockDim.x * 2] + src[idx + blockDim.x * 3] + 
                   src[idx + blockDim.x * 4] + src[idx + blockDim.x * 5] + src[idx + blockDim.x * 6] + src[idx + blockDim.x * 7]; 
    }

    __syncthreads();

    // in-place reduction in global memory
    // stride > 0 --> stride > 32, 因为每次迭代 >>1，即把最后6次迭代排除
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }
        // synchronize within threadblock
        __syncthreads();
    }

    // unrolling warp
    // warp 内的规约（reduction），也叫 warp unrolling reduction，其目的是：
    // 把一个 warp（最多 32 个线程）中共享内存 idata[tid] 中的值，规约成一个总和，最后存入 idata[0]。
    if (tid < 32) {
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid +  8];
        vmem[tid] += vmem[tid +  4];
        vmem[tid] += vmem[tid +  2];
        vmem[tid] += vmem[tid +  1];

        if (tid == 0) atomicAdd(dst, idata[0]);
    }
}
__global__ void  kReduceUnrollWarps8_shared(int *src, int *dst, unsigned int N)
{
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;
    __shared__ int sdata[256];
    
    // 阶段1：从全局内存加载到共享内存
    int sum = 0;
    if (idx + 7 * blockDim.x < N)
    {
        sum = src[idx] + src[idx + blockDim.x * 1] + src[idx + blockDim.x * 2] + 
              src[idx + blockDim.x * 3] + src[idx + blockDim.x * 4] + 
              src[idx + blockDim.x * 5] + src[idx + blockDim.x * 6] + 
              src[idx + blockDim.x * 7];
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    // 阶段2：在共享内存中进行规约
    #pragma unroll
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1)
    {
        if (tid < stride)
        {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // 阶段3：warp级规约
    // 允许 warp 内线程之间 直接交换寄存器数据；
    // 避免使用共享内存、无需同步；
    // mask 为活跃线程掩码，0xffffffff 表示所有 32 个线程都有效。
    // warp-level reduction using shuffle
    if (tid < 32)
    {
        #pragma unroll
        int val = sdata[tid];
        for (int stride = warpSize / 2; stride > 0; stride >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, stride);
        }
        
        if (tid == 0) atomicAdd(dst, val);
    }
}
void iReduceUnrollWarps8(int *src, int *dst, unsigned int n, int *sum) {
    dim3 blockSize(256);
    dim3 gridSize(CEIL(CEIL(n, blockSize.x), 8)); // 在grid维度取 1/8
    
    cudaMemset(dst, 0, sizeof(int));

    kReduceUnrollWarps8<<<gridSize, blockSize>>>(src, dst, n);
    CHECK(cudaDeviceSynchronize());

    *sum = 0;
    cudaMemcpy(sum, dst, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << *sum << ", ";
} 
void iReduceUnrollWarps8_shared(int *src, int *dst, unsigned int n, int *sum) {
    dim3 blockSize(256);
    dim3 gridSize(CEIL(CEIL(n, blockSize.x), 8)); // 在grid维度取 1/8
    
    cudaMemset(dst, 0, sizeof(int));

    kReduceUnrollWarps8_shared<<<gridSize, blockSize>>>(src, dst, n);
    CHECK(cudaDeviceSynchronize());

    *sum = 0;
    cudaMemcpy(sum, dst, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << *sum << ", ";
} 



template <unsigned int iBlockSize>
__global__ void kReduceCompleteUnroll(int *src, int *dst,
                                     unsigned int N)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = src + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < N)
    {
        int a1 = src[idx];
        int a2 = src[idx + blockDim.x];
        int a3 = src[idx + 2 * blockDim.x];
        int a4 = src[idx + 3 * blockDim.x];
        int b1 = src[idx + 4 * blockDim.x];
        int b2 = src[idx + 5 * blockDim.x];
        int b3 = src[idx + 6 * blockDim.x];
        int b4 = src[idx + 7 * blockDim.x];
        src[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();

    // in-place reduction and complete unroll
    if (iBlockSize >= 1024 && tid < 512) idata[tid] += idata[tid + 512];

    __syncthreads();

    if (iBlockSize >= 512 && tid < 256)  idata[tid] += idata[tid + 256];

    __syncthreads();

    if (iBlockSize >= 256 && tid < 128)  idata[tid] += idata[tid + 128];

    __syncthreads();

    if (iBlockSize >= 128 && tid < 64)   idata[tid] += idata[tid + 64];

    __syncthreads();

    // unrolling warp
    if (tid < 32)
    {
        volatile int *vsmem = idata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    if (tid == 0) atomicAdd(dst, idata[0]);
}
void iReduceCompleteUnroll(int *src, int *dst, unsigned int n, int *sum) {
    dim3 blockSize(256);
    dim3 gridSize(CEIL(CEIL(n, 8), blockSize.x)); // 在grid维度取 1/8
    
    cudaMemset(dst, 0, sizeof(int));

    kReduceCompleteUnroll<256><<<gridSize, blockSize>>>(src, dst, n);
    CHECK(cudaDeviceSynchronize());

    *sum = 0;
    cudaMemcpy(sum, dst, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << *sum << ", ";
} 



int main(int argc, char **argv)
{
    int repeat_times = 10;
    int *sum = (int *)malloc(sizeof(int));
    double iStart, iElaps;
    int size = 1 << 24;
    float total_time;
    std::cout << "______________________With array size: " << size << "______________________" << std::endl;

    size_t bytes = size * sizeof(int);
    int *h_src = (int *)malloc(bytes);
    int *cpu_src = (int *)malloc(bytes); 

    // initialize the array
    for (int i = 0; i < size; i++)
    {
        // random
        h_src[i] = (int)( rand() & 0xFF );
    }

    memcpy(cpu_src, h_src, bytes);

    // device memory
    int *d_src = nullptr, *d_dst = nullptr;
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_src), bytes));
    CHECK(cudaMalloc(reinterpret_cast<void **>(&d_dst), sizeof(int)));

    // cpu reduction
    iStart = seconds();
    int cpu_sum = recursiveReduce(cpu_src, size);
    iElaps = seconds() - iStart;
    std::cout << RED << "[host]: elapsed = " << iElaps * 1000 << " ms, " << "sum = " << cpu_sum << RESET << std::endl << std::endl;

    // gpu neighbored
    cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice);
    total_time = TIME_RECORD(repeat_times, ([&]{iReduceNeighbored(d_src, d_dst, size, sum);}));
    std::cout << RED << std::endl << "[device neighbored]: elapsed = " << total_time / repeat_times << " ms " << RESET << std::endl << std::endl;

    // gpu Interleaved
    cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice);
    total_time = TIME_RECORD(repeat_times, ([&]{iReduceInterleaved(d_src, d_dst, size, sum);}));
    std::cout << RED << std::endl << "[device interleaved]: elapsed = " << total_time / repeat_times << " ms" << RESET << std::endl << std::endl; 

    // gpu warp
    cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice);
    total_time = TIME_RECORD(repeat_times, ([&]{iReduceWarp(d_src, d_dst, size, sum);}));
    std::cout << RED << std::endl << "[device warp]: elapsed = " << total_time / repeat_times << " ms" << RESET << std::endl << std::endl; 

    // gpu warp vectorization
    cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice);
    total_time = TIME_RECORD(repeat_times, ([&]{iReduceWarpVec4(d_src, d_dst, size, sum);}));
    std::cout << RED << std::endl << "[device warp vectorization4]: elapsed = " << total_time / repeat_times << " ms" << RESET << std::endl << std::endl; 
    // gpu warp vectorization
    cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice);
    total_time = TIME_RECORD(repeat_times, ([&]{iReduceWarpVec4_v2(d_src, d_dst, size, sum);}));
    std::cout << RED << std::endl << "[device warp vectorization4_v2]: elapsed = " << total_time / repeat_times << " ms" << RESET << std::endl << std::endl; 

    // gpu Interleaved_shared
    cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice);
    total_time = TIME_RECORD(repeat_times, ([&]{iReduceInterleaved_shared(d_src, d_dst, size, sum);}));
    std::cout << RED << std::endl << "[device Interleaved_shared]: elapsed = " << total_time / repeat_times << RESET << " ms" << std::endl << std::endl; 

    // gpu Interleaved_shared_unrolling4
    cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice);
    total_time = TIME_RECORD(repeat_times, ([&]{iReduceInterleavedUnrolling4(d_src, d_dst, size, sum);}));
    std::cout << RED << std::endl << "[device Interleaved_shared_unrolling4]: elapsed = " << total_time / repeat_times << " ms" << RESET << std::endl << std::endl; 

    // gpu Interleaved_unrolling8
    cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice);
    total_time = TIME_RECORD(repeat_times, ([&]{iReduceInterleavedUnrolling8(d_src, d_dst, size, sum);}));
    std::cout << RED << std::endl << "[device Interleaved_unrolling8]: elapsed = " << total_time / repeat_times << " ms" << RESET << std::endl << std::endl; 

    // gpu Interleaved_unrolling8
    cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice);
    total_time = TIME_RECORD(repeat_times, ([&]{iReduceUnrollWarps8(d_src, d_dst, size, sum);}));
    std::cout << RED << std::endl << "[device Unroll8_warp]: elapsed = " << total_time / repeat_times << " ms" << RESET << std::endl << std::endl; 

    // gpu Interleaved_unrolling8_ shared
    cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice);
    total_time = TIME_RECORD(repeat_times, ([&]{iReduceUnrollWarps8_shared(d_src, d_dst, size, sum);}));
    std::cout << RED << std::endl << "[device Unroll8_warp_shared]: elapsed = " << total_time / repeat_times << " ms" << RESET << std::endl << std::endl; 

    // gpu Interleaved_Complete_Unroll
    cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice);
    total_time = TIME_RECORD(repeat_times, ([&]{iReduceCompleteUnroll(d_src, d_dst, size, sum);}));
    std::cout << RED << std::endl << "[device Complete Unroll]: elapsed = " << total_time / repeat_times << " ms" << RESET << std::endl << std::endl; 

    free(h_src);
    free(cpu_src);
    cudaFree(d_src);
    cudaFree(d_dst);

    return 1;
}
