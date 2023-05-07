
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include "opencv2/calib3d.hpp"
#include <iostream>
#include <bitset>

void calcCost_CUDA(unsigned char *census_l, unsigned char *census_r, int* pix_cost,  size_t rows, size_t cols);
