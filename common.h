#include <sys/time.h>

#ifndef _COMMON_H
#define _COMMON_H

// 向上取整
#define CEIL(x, y) (((x) + (y) - 1) / (y))

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
    }                                                                          \
}

#define TIME_RECORD(N, func)                                                                    \
    [&] {                                                                                       \
        float total_time = 0;                                                                   \
        for (int repeat = 0; repeat < N; ++repeat) {                                           \
            cudaEvent_t start, stop;                                                            \
            CHECK(cudaEventCreate(&start));                                                 \
            CHECK(cudaEventCreate(&stop));                                                  \
            CHECK(cudaEventRecord(start));                                                  \
            cudaEventQuery(start);                                                              \
            func();                                                                             \
            CHECK(cudaEventRecord(stop));                                                   \
            CHECK(cudaEventSynchronize(stop));                                              \
            float elapsed_time;                                                                 \
            CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));                        \
            if (repeat > 0) total_time += elapsed_time;                                         \
            CHECK(cudaEventDestroy(start));                                                 \
            CHECK(cudaEventDestroy(stop));                                                  \
        }                                                                                       \
        if (N == 0) return (float)0.0;                                                          \
        return total_time;                                                                      \
    }()


inline double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

#define RESET       "\033[0m"
#define RED         "\033[31m"
#define GREEN       "\033[32m"
#define YELLOW      "\033[33m"

#endif // _COMMON_H
