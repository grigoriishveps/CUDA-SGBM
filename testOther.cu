#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>
#include "opencv2/opencv.hpp"

typedef std::vector<std::vector<std::vector<unsigned long> > > cost_3d_array;
typedef std::vector<std::vector<std::vector<std::vector<unsigned long> > > > cost_4d_array;

using namespace std; 

class ScanLine {
public:
  ScanLine(int drow, int dcol, bool posdir) {
    this->drow = drow;
    this->dcol = dcol;
    this->posdir = posdir;
  }
  bool posdir;
  int drow, dcol;
};


#define D_LVL 64

class ScanLines8 {
public:
  ScanLines8() {
    this->path8.push_back(ScanLine(1, 1, true));
    this->path8.push_back(ScanLine(1, 0, true));
    this->path8.push_back(ScanLine(1, -1, true));
    this->path8.push_back(ScanLine(0, -1, false));
    this->path8.push_back(ScanLine(0, 1, true));
  }
  
  std::vector<ScanLine> path8;
};


void census_transform(cv::Mat &img, cv::Mat &census, size_t rows, size_t cols){
  unsigned char * const img_pnt_st = img.data;
  unsigned char * const census_pnt_st = census.data;

  for (int row=1; row<rows-1; row++) {
    for (int col=1; col<cols-1; col++) {

      unsigned char *center_pnt = img_pnt_st + cols*row + col;
      unsigned char val = 0;
      for (int drow=-1; drow<=1; drow++) {
        for (int dcol=-1; dcol<=1; dcol++) {
          
          if (drow == 0 && dcol == 0) {
            continue;
          }
          unsigned char tmp = *(center_pnt + dcol + drow*cols);
          val = (val + (tmp < *center_pnt ? 0 : 1)) << 1;        
        }
      }
      *(census_pnt_st + cols*row + col) = val;
    }
  }
  return;
}

unsigned char calc_hamming_dist(unsigned char val_l, unsigned char val_r) {

  unsigned char dist = 0;
  unsigned char d = val_l ^ val_r;

  while(d) {
    d = d & (d - 1);
    dist++;
  }
  return dist;  
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



unsigned short aggregate_cost(int row, int col, int depth, int path, cost_3d_array &pix_cost, cost_4d_array &agg_cost, size_t rows, size_t cols) {
  // Depth loop for current pix.
  unsigned long val0 = 0xFFFF, val1 = 0xFFFF, val2 = 0xFFFF, val3 = 0xFFFF;
  unsigned long min_prev_d = 0xFFFF;

ScanLines8 scanlines;
  int dcol = scanlines.path8[path].dcol;
  int drow = scanlines.path8[path].drow;

  // Pixel matching cost for current pix.
  unsigned long indiv_cost = pix_cost[row][col][depth];

  if (row - drow < 0 || rows <= row - drow || col - dcol < 0 || cols <= col - dcol) {
    agg_cost[path][row][col][depth] = indiv_cost;
    return agg_cost[path][row][col][depth];
  }

int p1=3, p2=20;

  // Depth loop for previous pix.
  for (int dd = 0; dd < D_LVL; dd++) {
    unsigned long prev = agg_cost[path][row-drow][col-dcol][dd];
    if (prev < min_prev_d) {
      min_prev_d = prev;
    }
    
    if (depth == dd) {
      val0 = prev;
    } else if (depth == dd + 1) {
      val1 = prev + p1;
    } else if (depth == dd - 1) {
      val2 = prev + p1;
    } else {
      unsigned long tmp = prev + p2;
      if (tmp < val3) {
        val3 = tmp;
      }            
    }
  }

  // Select minimum cost for current pix.
  agg_cost[path][row][col][depth] = std::min(std::min(std::min(val0, val1), val2), val3) + indiv_cost - min_prev_d;

  return agg_cost[path][row][col][depth];
}

void aggregate_cost_for_each_scanline(cost_3d_array &pix_cost, cost_4d_array &agg_cost, cost_3d_array &sum_cost, size_t rows, size_t cols)
{
  // Cost aggregation for positive direction.
  ScanLines8 scanlines;
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < cols; col++) {
      for (int path = 0; path < scanlines.path8.size(); path++) {
        if (scanlines.path8[path].posdir) {
          for (int d = 0; d < D_LVL; d++) {
            sum_cost[row][col][d] += aggregate_cost(row, col, d, path, pix_cost, agg_cost, rows, cols);
          }
        }
      }
    }
  }

  // Cost aggregation for negative direction.
  for (int row = rows - 1; 0 <= row; row--) {
    for (int col = cols - 1; 0 <= col; col--) {
      for (int path = 0; path < scanlines.path8.size(); path++) {
        if (!scanlines.path8[path].posdir) {
          for (int d = 0; d < D_LVL; d++) {
            sum_cost[row][col][d] += aggregate_cost(row, col, d, path, pix_cost, agg_cost, rows, cols);
          }
        }
      }
    }
  }
  return;
}

void calc_disparity(cost_3d_array &sum_cost, cv::Mat &disp_img, size_t rows, size_t cols){
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < cols; col++) {
      unsigned char min_depth = 0;
      unsigned long min_cost = sum_cost[row][col][min_depth];
      for (int d = 1; d < D_LVL; d++) {
        unsigned long tmp_cost = sum_cost[row][col][d];
        if (tmp_cost < min_cost) {
          min_cost = tmp_cost;
          min_depth = d;
        }
      }
      disp_img.at<unsigned char>(row, col) = min_depth;
    } 
  } 

  return;
}

void compute_disp(cv::Mat &left, cv::Mat &right)
{
  cost_3d_array pix_cost, sum_cost;
  size_t cols = left.cols, rows = left.rows;
  pix_cost.resize(rows);
  sum_cost.resize(rows);
  for (int row = 0; row < rows; row++) {
      pix_cost[row].resize(cols);
      sum_cost[row].resize(cols);
      for (int col = 0; col < cols; col++) {
          pix_cost[row][col].resize(D_LVL, 0x0000);
          sum_cost[row][col].resize(D_LVL, 0x0000);
      }
  }

  cost_4d_array agg_cost;
  ScanLines8 scanlines;
  int scanpath = scanlines.path8.size();
  agg_cost.resize(scanpath);
  for (int path = 0; path < scanpath; path++) {
      agg_cost[path].resize(rows);
      for (int row = 0; row < rows; row++) {
          agg_cost[path][row].resize(cols);
          for (int col = 0; col < cols; col++) {
              agg_cost[path][row][col].resize(D_LVL, 0x0000);
          }
      }
  }

  cv::Mat *census_l = new cv::Mat(rows, cols, CV_8UC1);
  cv::Mat *census_r = new cv::Mat(rows, cols, CV_8UC1);
  cv::Mat disparityMap, *disp_img = new cv::Mat(rows, cols, CV_8UC1);


  // 1. Census Transform.
  cout << "1. Census Transform" << endl;
  census_transform(left, *census_l, rows, cols);
  census_transform(right, *census_r, rows, cols);

  cv::imshow("Census Trans Left", *census_l);
  cv::imshow("Census Trans Right", *census_r);

  // 2. Calculate Pixel Cost.
  cout << "2. Calculate Pixel Cost." << endl;
  calc_pixel_cost(*census_l, *census_r, pix_cost, rows, cols);

  // 3. Aggregate Cost
  cout << "3. Aggregate Cost" << endl;
  aggregate_cost_for_each_scanline(pix_cost, agg_cost, sum_cost, rows, cols);

  // 4. Create Disparity Image.
  cout << "4. Create Disparity Image." << endl;
  calc_disparity(sum_cost, *disp_img, rows, cols);

  // Visualize Disparity Image.
  disparityMap = *disp_img;

    cv::Mat tmp;
    disparityMap.convertTo(tmp, CV_8U, 256.0/D_LVL);
    applyColorMap(tmp, tmp, cv::COLORMAP_JET);
    cv::imshow("Sgbm Result", tmp);
    cv::waitKey(0);

  return;
}

int main(int argc, char** argv) {
  std::cout << "SGBM Test Started!" << std::endl;
  std::string left_path = "leftImage1.png",
    right_path = "rightImage1.png";
  
  std::cout << "1. Open and load images" << std::endl;
  cv::Mat left = cv::imread(left_path, cv::IMREAD_GRAYSCALE);
  cv::Mat right = cv::imread(right_path, cv::IMREAD_GRAYSCALE);

  std::cout << "2. Initialize class" << std::endl;
  
  compute_disp(left, right);

  cout << "" << endl;
  while(1)
    {
        short key = (short) cv::waitKey();
        if( key == 27 || key == 'q' || key == 'Q') // 'ESC'
            break;
    }
  return 0;
}