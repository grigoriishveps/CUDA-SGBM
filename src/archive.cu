

// #define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

// #define checkCudaErrors(call)                                 \
//   do {                                                        \
//     cudaError_t err = call;                                   \
//     if (err != cudaSuccess) {                                 \
//       printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
//              cudaGetErrorString(err));                        \
//       exit(EXIT_FAILURE);                                     \
//     }                                                         \
//   } while (0)

// __global__ void matMult ( int * pix_cost, long * agg_cost,  size_t rows, size_t cols )
// {
//     int   bx  = blockIdx.x;     // block index
//     int   tx  = threadIdx.x;        // thread index

//     if( tx == 0) {
//       printf("Prepare path, %d\n", bx);
//     }

//     if (getDirectionPos(bx)) {
//       for (int row = 0; row < rows; row++) {
//         for (int col = D_LVL; col < cols; col++) {
//           aggregate_one_direction_on_CUDA(row, col, tx, bx, pix_cost, agg_cost, rows, cols);
//           __syncthreads();
//         }
//       }
//     } else {
//       for (int row = rows - 1; 0 <= row; row--) {
//         for (int col = cols - 1; D_LVL <= col; col--) {
//           aggregate_one_direction_on_CUDA(row, col, tx, bx, pix_cost, agg_cost, rows, cols);
//           __syncthreads();
//         }
//       }
//     }
// }

// __global__ void concatResCUDA ( long * agg_cost, int * res,  size_t rows, size_t cols ){
//   int   bx  = blockIdx.x;     // block index
//   int   by  = blockIdx.y;     // block index
//   int   tx  = threadIdx.x;    // thread index

//   unsigned long long  smallIndex = bx * cols * D_LVL + by * D_LVL + tx;
//   res[smallIndex] = 0;

//   for (int path = 0; path < PATHS; path++) {
//     res[smallIndex] += agg_cost[path * rows * cols * D_LVL + smallIndex];
//   }
// }


// void agregateCostCUDA(cost_3d_array &pix_cost, cost_3d_array &sum_cost, size_t rows, size_t cols) {
//     int numBytes = rows * cols * D_LVL * sizeof(int);
//     long bigNumBytes = rows * cols * D_LVL * PATHS * sizeof( long );

//     // allocate host memory
//     int * a = (int *) calloc(rows * cols * D_LVL, sizeof(int));
//     int * res = (int *) calloc(rows * cols * D_LVL, sizeof(int));
    
//     if (!a) {
//         printf("mem failure, exiting A \n");
//         exit(EXIT_FAILURE);
//     }

//     for ( int i = 0; i < rows; i++ ){
//       for ( int j = 0; j < cols; j++ ) 
//         for ( int x = 0; x < D_LVL; x++ ){
// 			      int	k = cols*i* D_LVL  + j * D_LVL + x;
			
//             a[k] = pix_cost[i][j][x];
//         }
//     }
        
//     // allocate device memory
//     int * adev = NULL;
//     long * bdev = NULL;
//     int * resCuda = NULL;
//     checkCudaErrors(cudaMalloc ( (void**)&adev, numBytes ));
//     checkCudaErrors(cudaMalloc ( (void**)&bdev, bigNumBytes ));
//     checkCudaErrors(cudaMalloc ( (void**)&resCuda, numBytes ));

//     // set kernel launch configuration
  
//     dim3 threads ( D_LVL );
//     dim3 blocks  ( rows, cols );
//     // create cuda event handles
//     cudaEvent_t start, stop;
//     float gpuTime = 0.0f;

//     checkCudaErrors(cudaEventCreate ( &start ));
//     checkCudaErrors(cudaEventCreate ( &stop ));
    
//     // asynchronously issue work to the GPU (all to stream 0)
//     cudaEventRecord ( start, 0 );
//     checkCudaErrors(cudaMemcpy( adev, a, numBytes, cudaMemcpyHostToDevice ));
    
//     matMult<<<PATHS, D_LVL>>> ( adev, bdev, rows, cols);

//     printf("Concat paths \n");
//     concatResCUDA<<<blocks, threads>>> ( bdev, resCuda, rows, cols);
    
//     checkCudaErrors(cudaMemcpy( res, resCuda, numBytes, cudaMemcpyDeviceToHost ));
//     cudaEventRecord ( stop, 0 );

//     cudaEventSynchronize ( stop );
//     cudaEventElapsedTime ( &gpuTime, start, stop );

//     for ( int i = 0; i < rows; i++ ){
//       for ( int j = 0; j < cols; j++ ) 
//         for ( int x = 0; x < D_LVL; x++ ){
// 			      long k = cols*i* D_LVL  + j * D_LVL + x;
//             sum_cost[i][j][x] = res[k]; 
//         }
//     }
    
//     // print the cpu and gpu times
//     printf("Time spent executing by the GPU: %.2f millseconds\n", gpuTime );

//     // release resources
//     cudaEventDestroy ( start );
//     cudaEventDestroy ( stop  );
//     checkCudaErrors(cudaFree( adev  ));
//     cudaFree  ( bdev  );
//     cudaFree  ( resCuda );

//     delete a;
//     delete res;
// }