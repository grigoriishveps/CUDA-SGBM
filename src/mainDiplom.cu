#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <stdio.h>
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/ximgproc.hpp"
#include <iostream>
#include <bitset>
#include "./helpers/helper.cuh"
#include "./calc_cost/calc_cost.cuh"
#include "./calc_disparity/calc_disparity.cuh"
#include "./calc_path/calc_path.cuh"
// #include "./old_calc_direction/old_calc_direction.cuh"

using namespace cv;
using namespace cv::ximgproc;
using namespace cv::cuda;
using namespace std;

#define D_LVL 64
#define WINDOW_WIDTH 9
#define WINDOW_HEIGHT 7
#define BLOCK_SIZE 1
#define PATHS 5
#define P1 2
#define P2 5
// #define P2 3


int main () {
    double solving_time, allTimeSolving = (double) getTickCount();
    double costTime, pathTime, disparityTime;


    Mat leftImage = cv::imread("./src/images/leftImage1.png",cv::IMREAD_GRAYSCALE);
    Mat rightImage = cv::imread("./src/images/rightImage1.png",cv::IMREAD_GRAYSCALE);

    size_t cols = leftImage.cols, rows = leftImage.rows;


    int *pix_cost = (int *) calloc(rows * cols * D_LVL, sizeof(int));
    int *sum_cost = (int *) calloc(rows * cols * D_LVL, sizeof(int));
    
    if (!pix_cost || !sum_cost) {
        printf("mem failure, exiting A \n");
        exit(EXIT_FAILURE);
    }

    cv::Mat *census_l = new cv::Mat(rows, cols, CV_8UC1);
    cv::Mat *census_r = new cv::Mat(rows, cols, CV_8UC1);
    cv::Mat disparityMap, *disp_img = new cv::Mat(rows, cols, CV_8UC1);


    cout << " Start timing"<< endl;
    solving_time = (double) getTickCount();
    // resize(leftImage, left_for_matcher, Size(),0.1,0.1, INTER_LINEAR_EXACT);
    // cvtColor(left_for_matcher,  left_for_matcher,  COLOR_BGR2GRAY);
    // left_for_matcher.convertTo(left_for_matcher, CV_16UC1);
    // leftImage.convertTo(leftImage, CV_16UC3);
    
    // short leftImageColors =

    // Mat splitResult[3];
    // split(leftImage, splitResult);
    // Mat leftImageR = splitResult[0];
    // Mat leftImageG = splitResult[1];
    // Mat leftImageB = splitResult[2];

    // imshow("Blue Channel", leftImageR);//showing Blue channel//
    // imshow("Green Channel", leftImageG);//showing Green channel//
    // imshow("Red Channel", leftImageB);
    // cout << leftImageR;

    // short** arrayLeftImageR = toMatArray(splitResult[0]);
    // short** arrayLeftImageG = toMatArray(splitResult[1]);
    // short** arrayLeftImageB = toMatArray(splitResult[2]);

    // short** arrayRightImageR = toMatArray(splitResult[0]);
    // short** arrayRightImageG = toMatArray(splitResult[1]);
    // short** arrayRightImageB = toMatArray(splitResult[2]);

    // std::bitset<32> **CSCTLeftImageR = toCSCT(arrayLeftImageR, leftImageR.rows, leftImageR.cols);
    // std::bitset<32> **CSCTLeftImageG = toCSCT(arrayLeftImageG, leftImageR.rows, leftImageR.cols);
    // std::bitset<32> **CSCTLeftImageB = toCSCT(arrayLeftImageB, leftImageR.rows, leftImageR.cols);

    // std::bitset<32> **CSCTRightImageR = toCSCT(arrayLeftImageR, leftImageR.rows, leftImageR.cols);
    // std::bitset<32> **CSCTRightImageG = toCSCT(arrayLeftImageG, leftImageR.rows, leftImageR.cols);
    // std::bitset<32> **CSCTRightImageB = toCSCT(arrayLeftImageB, leftImageR.rows, leftImageR.cols);

    // checkCSCT(leftImage);

    // TEST GRAYSCALE

    // // 1. Census Transform.
    // cout << "1. Census Transform" << endl;
    
    // short** arrayLeftImage = toMatArray(leftImage);
    // short** arrayRightImage = toMatArray(rightImage);

    // uint **CSCTLeftImage = toCSCT(arrayLeftImage, rows, cols);
    // uint **CSCTRightImage = toCSCT(arrayRightImage, rows, cols);

    // for (int row = 0; row < rows; row++) {
    //     delete [] arrayLeftImage[row];
    //     delete [] arrayRightImage[row];
    // }

    // delete[] arrayLeftImage;
    // delete[] arrayRightImage;

    // // 2. Calculate Pixel Cost.
    // cout << "2. Calculate Pixel Cost." << endl;
    // calc_pixel_cost(CSCTLeftImage, CSCTRightImage, pix_cost, rows, cols);
    
    // for (int row = 0; row < rows; row++) {
    //   delete[] CSCTLeftImage[row];
    //   delete[] CSCTRightImage[row];
    // }

    // delete[] CSCTLeftImage;
    // delete[] CSCTRightImage;

    // census_transform(leftImage, *census_l, rows, cols);
    // census_transform(rightImage, *census_r, rows, cols);
    // calc_pixel_cost(*census_l, *census_r, pix_cost, rows, cols);

    // aggregate_direction_cost(pix_cost, agg_cost, sum_cost, rows, cols);

  // cout << "1. Census Transform" << endl;


  // cv::imshow("Census Trans Left", *census_l);
  // cv::imshow("Census Trans Right", *census_r);

  // 2. Calculate Pixel Cost.
  census_transform(leftImage, *census_l, rows, cols);
  census_transform(rightImage, *census_r, rows, cols);
  cout << "1. Calculate Pixel Cost." << endl;
  costTime = (double) getTickCount();



  calcCost_CUDA(*census_l, *census_r, pix_cost, rows, cols);
  costTime = ((double)getTickCount() - costTime)/getTickFrequency();


  // 3. Aggregate Cost
  cout << "2. Aggregate Cost" << endl;
  pathTime = (double) getTickCount();
  // agregateCostCUDA(pix_cost, sum_cost, rows, cols);   // 1.9s
  optimized_agregateCostCUDA(pix_cost, sum_cost, rows, cols);   // 20ms
  // aggregate_direction_cost(pix_cost, sum_cost, rows, cols);   // OPENCV method
  pathTime = ((double)getTickCount() - pathTime)/getTickFrequency();


  // 4. Create Disparity Image.
  cout << "3. Create Disparity Image." << endl;
  disparityTime = (double) getTickCount();
  // calc_disparity(sum_cost, *disp_img);
  calc_disparity(sum_cost, *disp_img, rows, cols);
  disparityTime = ((double)getTickCount() - disparityTime)/getTickFrequency();
  disparityMap = *disp_img;

  // Visualize Disparity Image.
  disparityMap = *disp_img;
  disparityMap.convertTo(disparityMap, CV_8U, 256.0/D_LVL);
  applyColorMap(disparityMap, disparityMap, COLORMAP_JET);
  imshow("leftImage", disparityMap);

  // END TEST GRAYSCALE

  solving_time = ((double)getTickCount() - solving_time)/getTickFrequency();
  allTimeSolving = ((double)getTickCount() - allTimeSolving)/getTickFrequency();
  cout.precision(3);
  cout<<"Cost algorithm time: "<< costTime <<"s"<<endl; //230ms  // 120
  cout<<"Path algorithm time: "<< pathTime <<"s"<<endl;   //100ms  //48ms
  cout<<"Disparity algorithm time: "<< disparityTime <<"s"<<endl;  /// 36ms
  cout<<"Process time: "<<solving_time<<"s"<<endl;   //400ms    // 230ms
  cout<<"All run time: "<<allTimeSolving<<"s"<<endl;   // 400ms  // 233ms
  std::cout << "OK"<< std::endl;
  while(1)
  {
      short key = (short)waitKey();
      if( key == 27 || key == 'q' || key == 'Q') // 'ESC'
          break;
  }
  
  return 0;
}

