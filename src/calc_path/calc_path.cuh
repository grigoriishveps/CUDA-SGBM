#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <stdio.h>
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/ximgproc.hpp"
#include <iostream>
#include <bitset>

typedef int*** cost_3d_array;

__global__ void matMult ( int * pix_cost, long*  agg_cost,  size_t rows, size_t cols );
__global__ void optimised_concatResCUDA ( long*  agg_cost, int* res,  size_t rows, size_t cols );

__global__ void optimized_matMult_LEFT ( int * pix_cost, long* agg_cost,  size_t rows, size_t cols );
__global__ void optimized_matMult_RIGHT ( int * pix_cost, long* agg_cost,  size_t rows, size_t cols );
__global__ void optimized_matMult_TOP ( int * pix_cost, long* agg_cost,  size_t rows, size_t cols );
__global__ void clearResCUDA ( int * res,  size_t rows, size_t cols );

// __device__ int optimized_aggregate_one_direction_on_CUDA(int row, int col, int depth, int path, int* pix_cost, long* agg_cost, size_t rows, size_t cols);
void optimized_agregateCostCUDA(cost_3d_array &pix_cost, cost_3d_array &sum_cost, size_t rows, size_t cols);
void agregateCostCUDA(cost_3d_array &pix_cost, cost_3d_array &sum_cost, size_t rows, size_t cols);



