#include "calc_cost.cuh"

#define D_LVL 64

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

__device__ char calc_hamming_dist_CUDA(unsigned char val_l, unsigned char val_r) {
  unsigned char dist = 0;
  unsigned char d = val_l ^ val_r;

  while(d) {
    d = d & (d - 1);
    dist++;
  }
  return dist;  
}

// 13.8
__global__ void optimized_agregateCost_process (unsigned char *left, unsigned char *right, int *res, size_t rows_t, size_t cols_t )
{
    int row = blockIdx.x;  // block index
    int col = blockIdx.y;
    int depth = threadIdx.x;
    int rows = rows_t;
    int cols = cols_t;

    unsigned char val_l = static_cast<unsigned char>(left[row*cols + col]);
    
    unsigned char val_r = 0;
    int index  = row * cols* D_LVL + col * D_LVL + depth; 
    if (col - depth >= 0) {
      val_r = static_cast<unsigned char>(right[row*cols + col - depth]);
    }
    
    res[index] = calc_hamming_dist_CUDA(val_l, val_r);
}

// 15.5
// __global__ void optimized_agregateCost_process (unsigned char *left, unsigned char *right, int *res, size_t rows_t, size_t cols_t )
// {
//     int row = blockIdx.x;  // block index
//     int col = blockIdx.y;
//     int depth = threadIdx.x;
//     int rows = rows_t;
//     int cols = cols_t;

//     unsigned char val_l = static_cast<unsigned char>(left[row*cols + col]);
    
//     for (int d = 0; d < D_LVL; d++) {
//       unsigned char val_r = 0;
//       int index  = row * cols* D_LVL + col * D_LVL + d; 
//       if (col - d >= 0) {
//         val_r = static_cast<unsigned char>(right[row*cols + col - d]);
//       }
      
//       res[index] = calc_hamming_dist_CUDA(val_l, val_r);
//     }
// }


__host__ void calcCost_CUDA(cv::Mat &census_l, cv::Mat &census_r, int* pix_cost,  size_t rows, size_t cols) {
    printf("send");
    int numBytes = rows * cols * D_LVL * sizeof(int);
    int smallBytes = rows * cols * D_LVL * sizeof(unsigned char);
 
    // allocate device memory
    unsigned char * adev = NULL, *bdev = NULL;
    int * resCuda = NULL;

    checkCudaErrors(cudaMalloc ( (void**)&adev, smallBytes ));
    checkCudaErrors(cudaMalloc ( (void**)&bdev, smallBytes ));
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
      checkCudaErrors(cudaEventCreate ( &start ));
      checkCudaErrors(cudaEventCreate ( &stop ));
      
      // asynchronously issue work to the GPU (all to stream 0)
      cudaEventRecord ( start,  0 );
      checkCudaErrors(cudaMemcpy( adev, census_l.data, smallBytes, cudaMemcpyHostToDevice ));  // CAN ASYNC
      checkCudaErrors(cudaMemcpy( bdev, census_r.data, smallBytes, cudaMemcpyHostToDevice ));  // CAN ASYNC

      optimized_agregateCost_process<<<blocks, threads>>> ( adev, bdev, resCuda, rows, cols);

      checkCudaErrors(cudaMemcpy( pix_cost, resCuda, numBytes, cudaMemcpyDeviceToHost ));
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
    cudaFree  ( bdev );
    cudaFree  ( resCuda );
}







