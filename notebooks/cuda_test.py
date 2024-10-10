import os

# 设置NUMBA_CUDA_DRIVER环境变量，指向WSL2的libcuda.so
os.environ['NUMBA_CUDA_DRIVER'] = "/usr/lib/wsl/lib/libcuda.so"

# 继续您的CUDA操作
from numba import cuda
cuda.detect()
