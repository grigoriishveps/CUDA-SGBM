
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include "opencv2/calib3d.hpp"
#include <iostream>
#include <bitset>

__global__ void processAgregateCostCUDA (unsigned char *left, unsigned char *right, int *res, size_t rows_t, size_t cols_t );
