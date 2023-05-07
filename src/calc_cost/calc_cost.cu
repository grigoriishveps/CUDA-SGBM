#include "calc_cost.cuh"

#define D_LVL 64

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
__global__ void processAgregateCostCUDA  (unsigned char *left, unsigned char *right, int *res, size_t rows_t, size_t cols_t )
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
