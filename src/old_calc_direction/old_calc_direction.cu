
#include "old_calc_direction.cuh"
#include "../helpers/helper.cuh"

#define D_LVL 64
#define WINDOW_WIDTH 9
#define WINDOW_HEIGHT 7
#define BLOCK_SIZE 1
#define PATHS 5
#define P1 5
#define P2 20

unsigned short aggregate_one_direction(int row, int col, int depth, int path, cost_3d_array &pix_cost, cost_4d_array &agg_cost, size_t rows, size_t cols) {
  // Depth loop for current pix.
  unsigned long val0 = 0xFFFF, val1 = 0xFFFF, val2 = 0xFFFF, val3 = 0xFFFF;
  unsigned long min_prev_d = 0xFFFF;

  ScanLines8 scanlines;
  int dcol = scanlines.path8[path].dcol;
  int drow = scanlines.path8[path].drow;

  // Pixel matching cost for current pix.
  unsigned long indiv_cost = pix_cost[row][col][depth];

  if (row - drow < 0 || rows <= row - drow || col - dcol < D_LVL || cols <= col - dcol) {
    agg_cost[path][row][col][depth] = indiv_cost;
    return agg_cost[path][row][col][depth];
  }

  // Depth loop for previous pix.

  val0 = agg_cost[path][row-drow][col-dcol][depth];
  if (depth + 1 < D_LVL) {
    val1 = agg_cost[path][row-drow][col-dcol][depth + 1] + P1;
  }
  if (depth - 1 >= 0) {
    val2 = agg_cost[path][row-drow][col-dcol][depth - 1] + P1;
  }

  for (int dd = 0; dd < D_LVL; dd++) {
    unsigned long prev = agg_cost[path][row-drow][col-dcol][dd];
    if (prev < min_prev_d) {
      min_prev_d = prev;
    }
  }

  val3 = min_prev_d + P2;

  // Select minimum cost for current pix.
  agg_cost[path][row][col][depth] = std::min(std::min(std::min(val0, val1), val2), val3) + indiv_cost - min_prev_d;

  return agg_cost[path][row][col][depth];
}

void aggregate_direction_cost(cost_3d_array &pix_cost, cost_3d_array &sum_cost, size_t rows, size_t cols) {
    ScanLines8 scanlines;
    cost_4d_array agg_cost;
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

    // Cost aggregation for positive direction.

    for (int path = 0; path < scanlines.path8.size(); path++) {
      if (scanlines.path8[path].posdir) {
        for (int row = 0; row < rows; row++) {
          for (int col = D_LVL; col < cols; col++) {
            for (int d = 0; d < D_LVL; d++) {
              sum_cost[row][col][d] += aggregate_one_direction(row, col, d, path, pix_cost, agg_cost, rows, cols);
              sum_cost[row][col][d] = (int)sum_cost[row][col][d];
            }
          }
        }
      }
    }

    // Cost aggregation for negative direction.
    for (int path = 0; path < scanlines.path8.size(); path++) {
      if (!scanlines.path8[path].posdir) {
        for (int row = rows - 1; 0 <= row; row--) {
          for (int col = cols - 1; D_LVL <= col; col--) {
            for (int d = 0; d < D_LVL; d++) {
              sum_cost[row][col][d] += aggregate_one_direction(row, col, d, path, pix_cost, agg_cost, rows, cols);
              sum_cost[row][col][d] = (int)sum_cost[row][col][d];
            }
          }
        }
      }
    }
}
