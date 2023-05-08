#include "../helpers/helper.cuh"
#include "calc_path.cuh"

#include "../calc_disparity/calc_disparity.cuh"

#define D_LVL 64

#define P1 5
#define P2 10
// #define P1 25
// #define P2 49

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


__device__ int optimized_aggregate_direction_CUDA(
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

  int index  = row * cols* D_LVL + col * D_LVL + depth; 

  // Pixel matching cost for current pix.
  int indiv_cost = pix_cost[index];   // CAN OPTIMEZED

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

      int index  = row * cols* D_LVL + col * D_LVL + depthThread; 
      if (col == D_LVL) {
        agg_cost[index] = pix_cost[index];
        prevAgrArr[depthThread] = agg_cost[index];
      } else {
        prevAgrArr[depthThread] = optimized_aggregate_direction_CUDA(row, col, depthThread, pix_cost, agg_cost, rows, cols, min_prev_d, prevAgrArr);
      }

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

      int index  = row * cols* D_LVL + col * D_LVL + depthThread; 
      if (cols == col + 1) {
        agg_cost[index] = pix_cost[index];
        prevAgrArr[depthThread] = agg_cost[index];
      } else {
        prevAgrArr[depthThread] = optimized_aggregate_direction_CUDA(row, col, depthThread, pix_cost, agg_cost, rows, cols, min_prev_d, prevAgrArr);
      }

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

        int index  = row * cols* D_LVL + col * D_LVL + depthThread; 
        if (row == 0) {
          agg_cost[index] = pix_cost[index];
          prevAgrArr[depthThread] = agg_cost[index];
        } else {
          prevAgrArr[depthThread] = optimized_aggregate_direction_CUDA(row, col, depthThread, pix_cost, agg_cost, rows, cols, min_prev_d, prevAgrArr);
        }

        __syncthreads();
      }
    }
}

__global__ void optimized_matMult_LEFT_TOP ( int * pix_cost, int * agg_cost,  size_t rows_t, size_t cols_t )
{
    int block = blockIdx.x;     // block index
    int depth  = threadIdx.x;        // thread index
    __shared__ int min_prev_d;
    __shared__ int prevAgrArr[D_LVL];
    size_t rows = rows_t;
    size_t cols = cols_t;

    int row = (block < rows) ? block : 0;
    int col = (block < rows) ? 0 : block - rows + 1;

    if (col >= D_LVL) {
      while(row < rows && col < cols) {
        if(depth == 0) {
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
        int index  = row * cols* D_LVL + col * D_LVL + depth; 
        
        if (col == 0 || row == 0) {
          agg_cost[index] = pix_cost[index];
          prevAgrArr[depth] = agg_cost[index];
        } else {
          prevAgrArr[depth] = optimized_aggregate_direction_CUDA(row, col, depth, pix_cost, agg_cost, rows, cols, min_prev_d, prevAgrArr);
        }

        row++;
        col++;
        __syncthreads();
      };
    }
}


__global__ void optimized_matMult_RIGHT_TOP ( int * pix_cost, int * agg_cost,  size_t rows_t, size_t cols_t )
{
    int block = blockIdx.x;     // block index
    int depth  = threadIdx.x;        // thread index
    __shared__ int min_prev_d;
    __shared__ int prevAgrArr[D_LVL];
    size_t rows = rows_t;
    size_t cols = cols_t;

    int row = (block < cols) ? 0 : block - rows + 1;
    int col = (block < cols) ? block : cols - 1;

    if (col >= D_LVL) {
      while(row < rows && col >= 0) {
        if(depth == 0) {
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
        int index  = row * cols* D_LVL + col * D_LVL + depth; 
        
        if (cols == col + 1 || rows == row + 1) {
          agg_cost[index] = pix_cost[index];
          prevAgrArr[depth] = agg_cost[index];
        } else {
          prevAgrArr[depth] = optimized_aggregate_direction_CUDA(row, col, depth, pix_cost, agg_cost, rows, cols, min_prev_d, prevAgrArr);
        }

        row++;
        col++;
        __syncthreads();
      };
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
