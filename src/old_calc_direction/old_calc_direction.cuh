#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include "opencv2/calib3d.hpp"
#include <iostream>
#include <bitset>


typedef int*** cost_3d_array;
typedef std::vector<std::vector<std::vector<std::vector<int> > > > cost_4d_array;

void aggregate_direction_cost(cost_3d_array &pix_cost, cost_3d_array &sum_cost, size_t rows, size_t cols);

