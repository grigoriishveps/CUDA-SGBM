#include "../helpers/helper.cuh"
#include "calc_path.cuh"

#include "../calc_disparity/calc_disparity.cuh"

#define D_LVL 64
#define WINDOW_WIDTH 9
#define WINDOW_HEIGHT 7
#define BLOCK_SIZE 1
#define PATHS 5
#define P1 5
#define P2 20
// #define P1 2
// #define P2 5

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

#define checkCudaErrors(call)                                 \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)


__device__ long optimized_aggregate_LEFT_direction_CUDA(
  int row,
  int col,
  int depth,
  int* pix_cost,
  long* agg_cost,
  size_t rows,
  size_t cols,
  long min_prev_d,
  long *prevAgrArr
) {
  // Depth loop for current pix.
  long val0 = 0xFFFF, val1 = 0xFFFF, val2 = 0xFFFF, val3 = 0xFFFF;

  long index  = row * cols* D_LVL + col * D_LVL + depth;   // CAN OPTIMEZED

  // Pixel matching cost for current pix.
  long indiv_cost = pix_cost[index];   // CAN OPTIMEZED

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

__device__ long optimized_aggregate_RIGHT_direction_CUDA(
  int row,
  int col,
  int depth,
  int* pix_cost,
  long* agg_cost,
  size_t rows,
  size_t cols,
  long min_prev_d,
  long *prevAgrArr
) {
  // Depth loop for current pix.
  long val0 = 0xFFFF, val1 = 0xFFFF, val2 = 0xFFFF, val3 = 0xFFFF;

  long index  = row * cols* D_LVL + col * D_LVL + depth;   // CAN OPTIMEZED

  // Pixel matching cost for current pix.
  long indiv_cost = pix_cost[index];   // CAN OPTIMEZED

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

__device__ long optimized_aggregate_TOP_direction_CUDA(
  int row,
  int col,
  int depth,
  int* pix_cost,
  long* agg_cost,
  size_t rows,
  size_t cols,
  long min_prev_d,
  long *prevAgrArr
) {
  long val0 = 0xFFFF, val1 = 0xFFFF, val2 = 0xFFFF, val3 = 0xFFFF;

  long index  = row * cols* D_LVL + col * D_LVL + depth;   // CAN OPTIMEZED

  // Pixel matching cost for current pix.
  long indiv_cost = pix_cost[index];   // CAN OPTIMEZED

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


void optimized_agregateCostCUDA(cost_3d_array pix_cost, cost_3d_array sum_cost, size_t rows, size_t cols) {
    long numBytes = rows * cols * D_LVL * sizeof(int);
    long extraBytes = rows * cols * D_LVL * sizeof(long);

    // allocate host memory
    int *a = pix_cost;
    int *res = sum_cost;

    // allocate device memory
    int * adev = NULL;
    long * extraStore = NULL;
    int * resCuda = NULL;
    checkCudaErrors(cudaMalloc ( (void**)&adev, numBytes ));
    checkCudaErrors(cudaMalloc ( (void**)&extraStore, extraBytes ));
    checkCudaErrors(cudaMalloc ( (void**)&resCuda, numBytes ));

    // set kernel launch configuration
  
    dim3 threads ( D_LVL );
    dim3 blocks  ( rows, cols );
    // create cuda event handles
    cudaEvent_t start, stop;
    float gpuTime;
    float allRes = 0;
    int countCheck = 1;
    

    // TimeCheck
    for (int i = 0; i< countCheck ; i++) {
      gpuTime = 0.0f;
      clearResCUDA<<<blocks, threads>>> ( resCuda, rows, cols);
      checkCudaErrors(cudaEventCreate ( &start ));
      checkCudaErrors(cudaEventCreate ( &stop ));
      
      // asynchronously issue work to the GPU (all to stream 0)
      cudaEventRecord ( start, 0 );
      checkCudaErrors(cudaMemcpy( adev, a, numBytes, cudaMemcpyHostToDevice ));
      // checkCudaErrors(cudaMemcpy( adev, pix_cost, numBytes, cudaMemcpyHostToDevice ));
      // checkCudaErrors(cudaMemcpyAsync( adev, a, numBytes, cudaMemcpyHostToDevice, 16 ));
      
      optimized_matMult_LEFT<<<rows, D_LVL>>> ( adev, extraStore, rows, cols);
      optimised_concatResCUDA<<<blocks, threads>>> ( extraStore, resCuda, rows, cols);
      optimized_matMult_RIGHT<<<rows, D_LVL>>> ( adev, extraStore, rows, cols);
      optimised_concatResCUDA<<<blocks, threads>>> ( extraStore, resCuda, rows, cols);
      optimized_matMult_TOP<<<cols, D_LVL>>> ( adev, extraStore, rows, cols);
      optimised_concatResCUDA<<<blocks, threads>>> ( extraStore, resCuda, rows, cols);
      
      // optimised_concatResCUDA<<<blocks, threads>>> ( adev, resCuda, rows, cols);

      checkCudaErrors(cudaMemcpy( res, resCuda, numBytes, cudaMemcpyDeviceToHost ));
      cudaEventRecord ( stop, 0 );

      cudaEventSynchronize ( stop );
      cudaEventElapsedTime ( &gpuTime, start, stop );

      cudaEventDestroy ( start );
      cudaEventDestroy ( stop  );
      allRes += gpuTime;
       
      // printf("Time spent executing by the GPU: %.3f millseconds\n", gpuTime);
    }
    
    printf("Average Time spent executing by the GPU: %.3f millseconds for COUNT=%d \n", allRes / countCheck, countCheck);

    // release resources
 
    checkCudaErrors(cudaFree( adev  ));
    cudaFree  ( extraStore );
    cudaFree  ( resCuda );
}




__global__ void optimized_matMult_LEFT ( int * pix_cost, long * agg_cost,  size_t rows_t, size_t cols_t ) {
    int bx  = blockIdx.x;     // block index
    int depthThread  = threadIdx.x;        // thread index
    __shared__ long min_prev_d;
    __shared__ long prevAgrArr[D_LVL];
    int rows = rows_t;
    int cols = cols_t;
  
    // if( depthThread == 1 && bx==0) {
    //   printf("Prepare LEFT \n");
    // }

    int row = bx;

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

__global__ void optimized_matMult_RIGHT ( int * pix_cost, long * agg_cost,  size_t rows_t, size_t cols_t )
{
    int bx  = blockIdx.x;     // block index
    int depthThread  = threadIdx.x;        // thread index
    __shared__ long min_prev_d;
    __shared__ long prevAgrArr[D_LVL];
    int rows = rows_t;
    int cols = cols_t;

    // if(depthThread == 1 && bx==0) {
    //   printf("Prepare RIGHT \n");
    // }

    int row = bx;

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

__global__ void optimized_matMult_TOP ( int * pix_cost, long * agg_cost,  size_t rows_t, size_t cols_t )
{
    int bx  = blockIdx.x;     // block index
    int depthThread  = threadIdx.x;        // thread index
    __shared__ long min_prev_d;
    __shared__ long prevAgrArr[D_LVL];
    size_t rows = rows_t;
    size_t cols = cols_t;

    // if(depthThread == 1 && bx==0) {
    //   printf("Prepare TOP \n");
    // }

    int col = bx;

    if (bx >= D_LVL) {
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



__global__ void optimised_concatResCUDA (long * agg_cost, int* res,  size_t rows, size_t cols ){
  int   bx  = blockIdx.x;     // block index
  int   by  = blockIdx.y;     // block index
  int   tx  = threadIdx.x;    // thread index

  long smallIndex = bx * cols * D_LVL + by * D_LVL + tx;

  res[smallIndex] += agg_cost[smallIndex];
}

__global__ void clearResCUDA ( int * res,  size_t rows, size_t cols ){
  int   bx  = blockIdx.x;     // block index
  int   by  = blockIdx.y;     // block index
  int   tx  = threadIdx.x;    // thread index

  res[bx * cols * D_LVL + by * D_LVL + tx] = 0;
}
