#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <stdio.h>
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"

#include <iostream>
#include <bitset>

__global__ void matMult ( int * pix_cost, int*  agg_cost,  size_t rows, size_t cols );
__global__ void optimised_concatResCUDA ( int*  agg_cost, int* res,  size_t rows, size_t cols );

__global__ void optimized_matMult_LEFT ( int * pix_cost, int* agg_cost,  size_t rows, size_t cols );
__global__ void optimized_matMult_RIGHT ( int * pix_cost, int* agg_cost,  size_t rows, size_t cols );
__global__ void optimized_matMult_TOP ( int * pix_cost, int* agg_cost,  size_t rows, size_t cols );
__global__ void optimized_matMult_LEFT_TOP ( int * pix_cost, int * agg_cost,  size_t rows_t, size_t cols_t );
__global__ void optimized_matMult_RIGHT_TOP ( int * pix_cost, int * agg_cost,  size_t rows_t, size_t cols_t );
__global__ void clearResCUDA ( int * res,  size_t rows, size_t cols );

