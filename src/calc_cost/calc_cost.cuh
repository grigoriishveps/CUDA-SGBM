
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include "opencv2/calib3d.hpp"
#include <iostream>
#include <bitset>

typedef unsigned long*** cost_3d_array;

void calc_pixel_cost(unsigned int ** census_l, unsigned int **census_r, cost_3d_array &pix_cost, size_t rows, size_t cols);
void calc_pixel_cost(cv::Mat &census_l, cv::Mat &census_r, cost_3d_array &pix_cost, size_t rows, size_t cols);

