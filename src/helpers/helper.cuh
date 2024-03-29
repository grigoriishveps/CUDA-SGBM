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

void census_transform(cv::Mat &img, unsigned char *census, size_t rows, size_t cols);
uint** toCSCT (short** inArr, size_t rows, size_t cols);
short** toMatArray (cv::Mat mat);


