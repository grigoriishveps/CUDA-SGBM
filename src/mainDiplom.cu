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
#include "./helpers/helper.cuh"
#include "./calc_cost/calc_cost.cuh"
#include "./calc_disparity/calc_disparity.cuh"
#include "./calc_path/calc_path.cuh"

using namespace cv;

using namespace cv::cuda;
using namespace std;

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



__host__ void allProcessOnCUDA(unsigned char* census_l, unsigned char* census_r, int* pix_cost,  size_t rows, size_t cols) {
    int numBytes = rows * cols * D_LVL * sizeof(int);
    int smallBytes = rows * cols * D_LVL * sizeof(unsigned char);
    int extraBytes = rows * cols * D_LVL * sizeof(int);

    // allocate device memory
    unsigned char * adev = NULL, *bdev = NULL;
    int * extraStore = NULL;
    int * resCuda = NULL, *middleRes = NULL;

    checkCudaErrors(cudaMalloc ( (void**)&adev, smallBytes ));
    checkCudaErrors(cudaMalloc ( (void**)&bdev, smallBytes ));
    checkCudaErrors(cudaMalloc ( (void**)&middleRes, numBytes ));
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
        checkCudaErrors(cudaEventCreate ( &start ));
        checkCudaErrors(cudaEventCreate ( &stop ));
        
        // asynchronously issue work to the GPU (all to stream 0)
        cudaEventRecord ( start,  0 );
        checkCudaErrors(cudaMemcpy( adev, census_l, smallBytes, cudaMemcpyHostToDevice ));  // CAN ASYNC
        checkCudaErrors(cudaMemcpy( bdev, census_r, smallBytes, cudaMemcpyHostToDevice ));  // CAN ASYNC

        // COST
        processAgregateCostCUDA <<<blocks, threads>>> ( adev, bdev, middleRes, rows, cols);

        // PATH
        optimized_matMult_LEFT<<<rows, D_LVL>>> ( middleRes, extraStore, rows, cols);
        optimised_concatResCUDA<<<blocks, threads>>> (extraStore, resCuda, rows, cols);
        optimized_matMult_RIGHT<<<rows, D_LVL>>> (middleRes, extraStore, rows, cols);
        optimised_concatResCUDA<<<blocks, threads>>> (extraStore, resCuda, rows, cols);
        optimized_matMult_TOP<<<cols, D_LVL>>> (middleRes, extraStore, rows, cols);
        optimised_concatResCUDA<<<blocks, threads>>> (extraStore, resCuda, rows, cols);

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
    cudaFree  ( middleRes );
    cudaFree  ( extraStore );
    cudaFree  ( resCuda );
}


void calculateImageDisparity(cv::Mat &leftImage, cv::Mat &rightImage, cv::Mat *dispImg) {
    double costTime, disparityTime;
    size_t cols = leftImage.cols, rows = leftImage.rows;
    int *sum_cost = (int *) calloc(rows * cols * D_LVL, sizeof(int));

    if (!sum_cost) {
        printf("mem failure, exiting A \n");
        exit(EXIT_FAILURE);
    }

    unsigned char *census_l = (unsigned char *) calloc(rows * cols * D_LVL, sizeof(unsigned char));
    unsigned char *census_r = (unsigned char *) calloc(rows * cols * D_LVL, sizeof(unsigned char));

    // cout << "1. Census Transform" << endl;
    census_transform(leftImage, census_l, rows, cols);
    census_transform(rightImage, census_r, rows, cols);

    // 2. Calculate Pixel Cost.
    // 3. Aggregate Cost
    //One CUDA operation
    costTime = (double) getTickCount();
    allProcessOnCUDA(census_l, census_r, sum_cost, rows, cols);
    costTime = ((double)getTickCount() - costTime)/getTickFrequency();

    // 4. Create Disparity Image.
    cout << "3. Create Disparity Image." << endl;
    disparityTime = (double) getTickCount();
    calc_disparity(sum_cost, *dispImg, rows, cols);
    disparityTime = ((double)getTickCount() - disparityTime)/getTickFrequency();

    cout<<"Cost algorithm time: "<< costTime <<"s"<<endl;  // 120ms
    cout<<"Disparity algorithm time: "<< disparityTime <<"s"<<endl;  // 36ms

    free(sum_cost);
    free(census_l);
    free(census_r);
}


int main () {
    double solving_time, allTimeSolving = (double) getTickCount();

    Mat leftImage = cv::imread("./src/images/leftImage1.png",cv::IMREAD_GRAYSCALE);
    Mat rightImage = cv::imread("./src/images/rightImage1.png",cv::IMREAD_GRAYSCALE);
    // Mat leftImage = cv::imread("./src/images/appleLeft.png",cv::IMREAD_GRAYSCALE);
    // Mat rightImage = cv::imread("./src/images/appleRight.png",cv::IMREAD_GRAYSCALE);

    size_t cols = leftImage.cols, rows = leftImage.rows;
    cv::Mat disparityMap, *dispImg = new cv::Mat(rows, cols, CV_8UC1);

    cout.precision(3);
    cout << " Start timing"<< endl;
    solving_time = (double) getTickCount();
    // resize(leftImage, left_for_matcher, Size(),0.1,0.1, INTER_LINEAR_EXACT);
    // cvtColor(left_for_matcher,  left_for_matcher,  COLOR_BGR2GRAY);
    // left_for_matcher.convertTo(left_for_matcher, CV_16UC1);
    // leftImage.convertTo(leftImage, CV_16UC3);
    
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

    calculateImageDisparity(leftImage, rightImage, dispImg);
  


    // Visualize Disparity Image.
    disparityMap = *dispImg;
    disparityMap.convertTo(disparityMap, CV_8U, 256.0/D_LVL);
    applyColorMap(disparityMap, disparityMap, COLORMAP_JET);
    imshow("leftImage", disparityMap);

    // END TEST GRAYSCALE

    solving_time = ((double)getTickCount() - solving_time)/getTickFrequency();
    allTimeSolving = ((double)getTickCount() - allTimeSolving)/getTickFrequency();

    cout<<"Process time: "<<solving_time<<"s"<<endl;     // 179ms
    cout<<"All run time: "<<allTimeSolving<<"s"<<endl;   // 184ms
    std::cout << "OK"<< std::endl;

    free(dispImg);

    while(1)
    {
        short key = (short)waitKey();
        if( key == 27 || key == 'q' || key == 'Q') // 'ESC'
            break;
    }

    return 0;
}

