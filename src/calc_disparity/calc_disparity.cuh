
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include "opencv2/calib3d.hpp"
#include <iostream>
#include <bitset>

__global__ void calcDisparity(int*sumCost, uchar* disparityMap, size_t cols);