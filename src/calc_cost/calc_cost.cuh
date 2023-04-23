
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include "opencv2/calib3d.hpp"
#include <iostream>
#include <bitset>

typedef int* cost_3d_array;

void calcCost_CUDA(cv::Mat &census_l, cv::Mat &census_r, cost_3d_array &pix_cost,  size_t rows, size_t cols);
