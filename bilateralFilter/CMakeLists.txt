cmake_minimum_required(VERSION 3.22)
project(npp_bilateral_filter LANGUAGES CXX CUDA)

# 找到 CUDA Toolkit 和 NPP
find_package(CUDAToolkit REQUIRED)

# 指定 CUDA 版本，确保启用 NPP
message(STATUS "CUDA Toolkit found: ${CUDAToolkit_VERSION}")

# 设置 CUDA 编译标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 可执行文件
add_executable(npp_bilateral
    src/main.cpp
)

# 链接 NPP 库
target_link_libraries(npp_bilateral
    PUBLIC
    CUDA::cudart 
    CUDA::nppc
    CUDA::nppial
    CUDA::nppicc
    CUDA::nppif
)

# 对于 Ubuntu 上，如果找不到静态模块，可改用动态库
# target_link_libraries(npp_bilateral PUBLIC
#     nppc nppial nppicc nppif
# )

# 包含路径（可选）
target_include_directories(npp_bilateral PUBLIC ${CUDAToolkit_INCLUDE_DIRS})

# 编译选项
target_compile_options(npp_bilateral
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC>
)
