#include "../helpers/helper.cuh"
#include "calc_path.cuh"

#include "../calc_disparity/calc_disparity.cuh"

#define D_LVL 64
#define PATHS 5
#define P1 20
#define P2 30
// #define P1 24
// #define P2 48  //96

#define checkCudaErrors(call)                                 \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)


__device__ int optimized_aggregate_LEFT_direction_CUDA(
  int row,
  int col,
  int depth,
  int* pix_cost,
  int* agg_cost,
  size_t rows,
  size_t cols,
  int min_prev_d,
  int *prevAgrArr
) {
  // Depth loop for current pix.
  int val0 = 0xFFFF, val1 = 0xFFFF, val2 = 0xFFFF, val3 = 0xFFFF;

  int index  = row * cols* D_LVL + col * D_LVL + depth;   // CAN OPTIMEZED

  // Pixel matching cost for current pix.
  int indiv_cost = pix_cost[index];   // CAN OPTIMEZED

  if (col == D_LVL) {
    agg_cost[index] = indiv_cost;
    return agg_cost[index];
  }

  val0 = prevAgrArr[depth];
  if (depth + 1 < D_LVL) {
    val1 = prevAgrArr[depth + 1] + P1;
  }
  if (depth - 1 >= 0) {
    val2 = prevAgrArr[depth - 1] + P1;
  }

  val3 = min_prev_d + P2;

  // Select minimum cost for current pix.
  agg_cost[index] = min(min(min(val0, val1), val2), val3) + indiv_cost - min_prev_d;

  return agg_cost[index];
}

__device__ int optimized_aggregate_RIGHT_direction_CUDA(
  int row,
  int col,
  int depth,
  int* pix_cost,
  int* agg_cost,
  size_t rows,
  size_t cols,
  int min_prev_d,
  int *prevAgrArr
) {
  // Depth loop for current pix.
  int val0 = 0xFFFF, val1 = 0xFFFF, val2 = 0xFFFF, val3 = 0xFFFF;

  int index  = row * cols* D_LVL + col * D_LVL + depth;   // CAN OPTIMEZED

  // Pixel matching cost for current pix.
  int indiv_cost = pix_cost[index];   // CAN OPTIMEZED

  if (cols == col + 1) {
    agg_cost[index] = indiv_cost;
    return agg_cost[index];
  }

  val0 = prevAgrArr[depth];
  if (depth + 1 < D_LVL) {
    val1 = prevAgrArr[depth + 1] + P1;
  }
  if (depth - 1 >= 0) {
    val2 = prevAgrArr[depth - 1] + P1;
  }

  val3 = min_prev_d + P2;

  // Select minimum cost for current pix.
  agg_cost[index] = min(min(min(val0, val1), val2), val3) + indiv_cost - min_prev_d;

  return agg_cost[index];
}

__device__ int optimized_aggregate_TOP_direction_CUDA(
  int row,
  int col,
  int depth,
  int* pix_cost,
  int* agg_cost,
  size_t rows,
  size_t cols,
  int min_prev_d,
  int *prevAgrArr
) {
  int val0 = 0xFFFF, val1 = 0xFFFF, val2 = 0xFFFF, val3 = 0xFFFF;

  int index  = row * cols* D_LVL + col * D_LVL + depth;   // CAN OPTIMEZED

  // Pixel matching cost for current pix.
  int indiv_cost = pix_cost[index];   // CAN OPTIMEZED

  if (row == 0) {
    agg_cost[index] = indiv_cost;
    return agg_cost[index];
  }

  val0 = prevAgrArr[depth];
  if (depth + 1 < D_LVL) {
    val1 = prevAgrArr[depth + 1] + P1;
  }
  if (depth - 1 >= 0) {
    val2 = prevAgrArr[depth - 1] + P1;
  }

  val3 = min_prev_d + P2;

  // Select minimum cost for current pix.
  agg_cost[index] = min(min(min(val0, val1), val2), val3) + indiv_cost - min_prev_d;

  return agg_cost[index];
}


__global__ void optimized_matMult_LEFT ( int * pix_cost, int * agg_cost,  size_t rows_t, size_t cols_t ) {
    int row  = blockIdx.x;     // block index
    int depthThread  = threadIdx.x;        // thread index
    __shared__ int min_prev_d;
    __shared__ int prevAgrArr[D_LVL];
    int rows = rows_t;
    int cols = cols_t;

    for (int col = D_LVL; col < cols; col++) {
      // Depth loop for previous pix.

      if(depthThread == 0) {
        min_prev_d = 0xFFFF;
      
        if (col != D_LVL) {
          for (int dd = 1; dd < D_LVL; dd++) {
            if (prevAgrArr[dd] < min_prev_d) {
              min_prev_d = prevAgrArr[dd];
            }
          }
        }
      }

      __syncthreads();
      prevAgrArr[depthThread] = optimized_aggregate_LEFT_direction_CUDA(row, col, depthThread, pix_cost, agg_cost, rows, cols, min_prev_d, prevAgrArr);
      __syncthreads();
    }
}

__global__ void optimized_matMult_RIGHT ( int * pix_cost, int * agg_cost,  size_t rows_t, size_t cols_t )
{
    int row  = blockIdx.x;     // block index
    int depthThread  = threadIdx.x;        // thread index
    __shared__ int min_prev_d;
    __shared__ int prevAgrArr[D_LVL];
    int rows = rows_t;
    int cols = cols_t;

    for (int col = cols - 1; col >= D_LVL; col--) {
      // Depth loop for previous pix.

      if(depthThread == 0) {
        min_prev_d = 0xFFFF;
        
        for (int dd = 1; dd < D_LVL; dd++) {
          if (prevAgrArr[dd] < min_prev_d) {
            min_prev_d = prevAgrArr[dd];
          }
        }
      }

      __syncthreads();
      prevAgrArr[depthThread] = optimized_aggregate_RIGHT_direction_CUDA(row, col, depthThread, pix_cost, agg_cost, rows, cols, min_prev_d, prevAgrArr);
      __syncthreads();
    }
}

__global__ void optimized_matMult_TOP ( int * pix_cost, int * agg_cost,  size_t rows_t, size_t cols_t )
{
    int col  = blockIdx.x;     // block index
    int depthThread  = threadIdx.x;        // thread index
    __shared__ int min_prev_d;
    __shared__ int prevAgrArr[D_LVL];
    size_t rows = rows_t;
    size_t cols = cols_t;

    if (col >= D_LVL) {
      for (int row = 0; row < rows; row++) {
        if(depthThread == 0) {
          min_prev_d = 0xFFFF;
        
          // Depth loop for previous pix.
          if ( row != 0 ) {
            for (int dd = 1; dd < D_LVL; dd++) {
              if (prevAgrArr[dd] < min_prev_d) {
                min_prev_d = prevAgrArr[dd];
              }
            }
          }
        }

        __syncthreads();
        prevAgrArr[depthThread] = optimized_aggregate_TOP_direction_CUDA(row, col, depthThread, pix_cost, agg_cost, rows, cols, min_prev_d, prevAgrArr);
        __syncthreads();
      }
    }
}



__global__ void optimised_concatResCUDA (int * agg_cost, int* res,  size_t rows, size_t cols ){
  int   bx  = blockIdx.x;     // block index
  int   by  = blockIdx.y;     // block index
  int   tx  = threadIdx.x;    // thread index

  int smallIndex = bx * cols * D_LVL + by * D_LVL + tx;

  res[smallIndex] += agg_cost[smallIndex];
}

__global__ void clearResCUDA ( int * res,  size_t rows, size_t cols ){
  int   bx  = blockIdx.x;     // block index
  int   by  = blockIdx.y;     // block index
  int   tx  = threadIdx.x;    // thread index

  res[bx * cols * D_LVL + by * D_LVL + tx] = 0;
}
