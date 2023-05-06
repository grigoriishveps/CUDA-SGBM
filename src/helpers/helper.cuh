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


extern __device__ int getDirectionY(size_t path);
extern __device__ int getDirectionX(size_t path);
extern __device__ int getDirectionPos(size_t path);

class ScanLine {
  public:
    ScanLine(int drow, int dcol, bool posdir);
    bool posdir;
    int drow, dcol;
};

class ScanLines8 {
  public:
    ScanLines8();
    std::vector<ScanLine> path8;
};

void census_transform(cv::Mat &img, cv::Mat &census, size_t rows, size_t cols);
uint** toCSCT (short** inArr, size_t rows, size_t cols);
short** toMatArray (cv::Mat mat);


