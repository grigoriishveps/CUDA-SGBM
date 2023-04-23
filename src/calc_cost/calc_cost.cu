#include "calc_cost.cuh"

#define D_LVL 64

unsigned char calc_hamming_dist(unsigned char val_l, unsigned char val_r) {

  unsigned char dist = 0;
  unsigned char d = val_l ^ val_r;

  while(d) {
    d = d & (d - 1);
    dist++;
  }
  return dist;  
}


void calc_pixel_cost(unsigned int ** census_l, unsigned int **census_r, cost_3d_array &pix_cost, size_t rows, size_t cols) {
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < cols; col++) {
      for (int d = 0; d < D_LVL; d++) {
        if (col - d < 0) {
            pix_cost[row][col][d] = 0;
            continue;
        }
        uint val_l = (unsigned short) census_l[row][col - d];
        uint val_r = (unsigned short) census_r[row][col];
        
        pix_cost[row][col][d] = calc_hamming_dist(val_l, val_r);
      }
    }
  }
}

void calc_pixel_cost(cv::Mat &census_l, cv::Mat &census_r, cost_3d_array &pix_cost, size_t rows, size_t cols) {

  unsigned char * const census_l_ptr_st = census_l.data;
  unsigned char * const census_r_ptr_st = census_r.data;

  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < cols; col++) {
      unsigned char val_l = static_cast<unsigned char>(*(census_l_ptr_st + row*cols + col));
      for (int d = 0; d < D_LVL; d++) {
        unsigned char val_r = 0;
        if (col - d >= 0) {
          val_r = static_cast<unsigned char>(*(census_r_ptr_st + row*cols + col - d));
        }
        pix_cost[row][col][d] = calc_hamming_dist(val_l, val_r);
      }
    }
  }
}






