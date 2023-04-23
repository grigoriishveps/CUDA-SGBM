
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include "opencv2/calib3d.hpp"
#include <iostream>
#include <bitset>

typedef int* cost_3d_array;

void calc_disparity(cost_3d_array &sum_cost, cv::Mat &disp_img, size_t rows, size_t cols);
