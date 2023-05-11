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
#include "./calc_cost/calc_cost_and_disparity.cuh"
#include "./calc_path/calc_path.cuh"

using namespace cv;

using namespace cv::cuda;
using namespace std;

#define D_LVL 64

#define checkCudaErrors(call)                                 \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)



__host__ void allProcessOnCUDA(
    unsigned char* census_l_R, unsigned char* census_l_G, unsigned char* census_l_B,
    unsigned char* census_r_R, unsigned char* census_r_G, unsigned char* census_r_B,
    uchar* disparityMap,
    size_t rows, size_t cols
) {
    int numBytes = rows * cols * D_LVL * sizeof(int);
    int smallBytes = rows * cols * D_LVL * sizeof(unsigned char);

    // allocate device memory
    unsigned char * adev = NULL, *bdev = NULL, *disparityRes = NULL;
    int * extraStore = NULL;
    int * resCuda = NULL, *middleRes = NULL;

    checkCudaErrors(cudaMalloc ( (void**)&adev, smallBytes ));
    checkCudaErrors(cudaMalloc ( (void**)&bdev, smallBytes ));
    checkCudaErrors(cudaMalloc ( (void**)&middleRes, numBytes ));
    checkCudaErrors(cudaMalloc ( (void**)&extraStore, numBytes ));
    checkCudaErrors(cudaMalloc ( (void**)&resCuda, numBytes  ));
    checkCudaErrors(cudaMalloc ( (void**)&disparityRes, smallBytes));

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

        clearResCUDA<<<blocks, threads>>> (middleRes, rows, cols);
        clearResCUDA<<<blocks, threads>>> (resCuda, rows, cols);
        clearResCUDA<<<blocks, threads>>> (disparityRes, rows, cols);
        // COST
        checkCudaErrors(cudaMemcpy( adev, census_l_R, smallBytes, cudaMemcpyHostToDevice ));
        checkCudaErrors(cudaMemcpy( bdev, census_r_R, smallBytes, cudaMemcpyHostToDevice ));  
        calculateInitialCostCUDA <<<blocks, threads>>> ( adev, bdev, extraStore, rows, cols);
        optimisedConcatResCUDA<<<blocks, threads>>> (extraStore, middleRes, rows, cols);

        checkCudaErrors(cudaMemcpy( adev, census_l_G, smallBytes, cudaMemcpyHostToDevice ));
        checkCudaErrors(cudaMemcpy( bdev, census_r_G, smallBytes, cudaMemcpyHostToDevice ));  
        calculateInitialCostCUDA <<<blocks, threads>>> ( adev, bdev, extraStore, rows, cols);
        optimisedConcatResCUDA<<<blocks, threads>>> (extraStore, middleRes, rows, cols);
    
        checkCudaErrors(cudaMemcpy( adev, census_l_B, smallBytes, cudaMemcpyHostToDevice ));
        checkCudaErrors(cudaMemcpy( bdev, census_r_B, smallBytes, cudaMemcpyHostToDevice ));  
        calculateInitialCostCUDA <<<blocks, threads>>> ( adev, bdev, extraStore, rows, cols);
        optimisedConcatResCUDA<<<blocks, threads>>> (extraStore, middleRes, rows, cols);


        // PATH
        calculatePathLeft<<<rows, D_LVL>>> ( middleRes, extraStore, rows, cols);
        optimisedConcatResCUDA<<<blocks, threads>>> (extraStore, resCuda, rows, cols);
        calculatePathRight<<<rows, D_LVL>>> (middleRes, extraStore, rows, cols);
        optimisedConcatResCUDA<<<blocks, threads>>> (extraStore, resCuda, rows, cols);
        calculatePathTop<<<cols, D_LVL>>> (middleRes, extraStore, rows, cols);
        optimisedConcatResCUDA<<<blocks, threads>>> (extraStore, resCuda, rows, cols);
        calculatePathBackslash<<<cols + rows - 1, D_LVL>>> (middleRes, extraStore, rows, cols);
        optimisedConcatResCUDA<<<blocks, threads>>> (extraStore, resCuda, rows, cols);
        calculatePathSlash<<<cols + rows - 1, D_LVL>>> (middleRes, extraStore, rows, cols);
        optimisedConcatResCUDA<<<blocks, threads>>> (extraStore, resCuda, rows, cols);


        //Disparyty
        calculateDisparityCUDA<<<blocks, 1>>> (resCuda, disparityRes, cols);

        checkCudaErrors(cudaMemcpy( disparityMap, disparityRes, smallBytes, cudaMemcpyDeviceToHost ));
        
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
    cudaFree  ( disparityRes );
}


cv::Mat* calculateImageDisparity(cv::Mat &leftImage, cv::Mat &rightImage) {
    double costTime;
    size_t cols = leftImage.cols, rows = leftImage.rows;
    uchar *disparityMap = (uchar *) calloc(rows * cols * D_LVL, sizeof(uchar));

    if (!disparityMap) {
        printf("mem failure, exiting A \n");
        exit(EXIT_FAILURE);
    }


    Mat splitResult[3], splitResultRight[3];
    split(leftImage, splitResult);
    Mat leftImageR = splitResult[0];
    Mat leftImageG = splitResult[1];
    Mat leftImageB = splitResult[2];
    split(rightImage, splitResultRight);
    Mat rightImageR = splitResultRight[0];
    Mat rightImageG = splitResultRight[1];
    Mat rightImageB = splitResultRight[2];

    // imshow("leftImageR", leftImageR);
    // imshow("leftImageG", leftImageG);
    // imshow("leftImageB", leftImageB);

    unsigned char *census_l_R = (unsigned char *) calloc(rows * cols * D_LVL, sizeof(unsigned char));
    unsigned char *census_l_G = (unsigned char *) calloc(rows * cols * D_LVL, sizeof(unsigned char));
    unsigned char *census_l_B = (unsigned char *) calloc(rows * cols * D_LVL, sizeof(unsigned char));
    unsigned char *census_r_R = (unsigned char *) calloc(rows * cols * D_LVL, sizeof(unsigned char));
    unsigned char *census_r_G = (unsigned char *) calloc(rows * cols * D_LVL, sizeof(unsigned char));
    unsigned char *census_r_B = (unsigned char *) calloc(rows * cols * D_LVL, sizeof(unsigned char));

    // 1. Census Transform"
    census_transform(leftImageR, census_l_R, rows, cols);
    census_transform(leftImageG, census_l_G, rows, cols);
    census_transform(leftImageB, census_l_B, rows, cols);
    census_transform(rightImageR, census_r_R, rows, cols);
    census_transform(rightImageG, census_r_G, rows, cols);
    census_transform(rightImageB, census_r_B, rows, cols);

    // 2. Calculate Pixel Cost.
    // 3. Aggregate Cost
    // 4. Create Disparity Image.
    //One CUDA operation
    costTime = (double) getTickCount();
    allProcessOnCUDA(census_l_R, census_l_G, census_l_B, census_r_R, census_r_G, census_r_B, disparityMap, rows, cols);
    costTime = ((double)getTickCount() - costTime)/getTickFrequency();

    cout<<"Cost algorithm time: "<< costTime <<"s"<<endl;  // 123ms

    free(census_l_R);
    free(census_r_G);
    free(census_l_B);
    free(census_r_R);
    free(census_l_G);
    free(census_r_B);
    
    return new Mat(rows, cols, CV_8UC1, disparityMap);
}


int main () {
    double solving_time, allTimeSolving = (double) getTickCount();
    // Mat leftImage = cv::imread("./src/images/leftImage1.png",cv::IMREAD_GRAYSCALE);
    // Mat rightImage = cv::imread("./src/images/rightImage1.png",cv::IMREAD_GRAYSCALE);
    // Mat leftImage = cv::imread("./src/images/leftImage1.png",cv::IMREAD_COLOR);
    // Mat rightImage = cv::imread("./src/images/rightImage1.png",cv::IMREAD_COLOR);
    // Mat leftImage = cv::imread("./src/images/appleLeft.jpg");
    // Mat rightImage = cv::imread("./src/images/appleRight.jpg");
    // Mat leftImage = cv::imread("./src/images/warLeft.jpg",cv::IMREAD_COLOR);
    // Mat rightImage = cv::imread("./src/images/warRight.jpg",cv::IMREAD_COLOR);
    Mat leftImage = cv::imread("./src/images/warLeft.jpg");
    Mat rightImage = cv::imread("./src/images/warRight.jpg");
    // Mat leftImage = cv::imread("./src/images/warLeft.jpg",cv::IMREAD_GRAYSCALE);
    // Mat rightImage = cv::imread("./src/images/warRight.jpg",cv::IMREAD_GRAYSCALE);

    // imshow("leftImage", leftImage);
    // imshow("rightImage", rightImage);

    cv::Mat *disparityMap;

    cout.precision(3);
    cout << " Start timing"<< endl;
    solving_time = (double) getTickCount();

    // resize(leftImage, left_for_matcher, Size(),0.1,0.1, INTER_LINEAR_EXACT);
    // cvtColor(left_for_matcher,  left_for_matcher,  COLOR_BGR2GRAY);
    // left_for_matcher.convertTo(left_for_matcher, CV_16UC1);
    // leftImage.convertTo(leftImage, CV_16UC3);
    disparityMap = calculateImageDisparity(leftImage, rightImage);




    // END TEST GRAYSCALE
    solving_time = ((double)getTickCount() - solving_time)/getTickFrequency();
 

    // Visualize Disparity Image.
    disparityMap->convertTo(*disparityMap, CV_8U, 256.0/D_LVL);
    applyColorMap(*disparityMap, *disparityMap, COLORMAP_JET);
    imshow("disparityMap", *disparityMap);

    allTimeSolving = ((double)getTickCount() - allTimeSolving)/getTickFrequency();
    cout<<"Process time: "<<solving_time<<"s"<<endl;     // 179ms
    cout<<"All run time: "<<allTimeSolving<<"s"<<endl;   // 184ms
    std::cout << "OK"<< std::endl;

    free(disparityMap);

    while(1)
    {
        short key = (short)waitKey();
        if( key == 27 || key == 'q' || key == 'Q') // 'ESC'
            break;
    }

    return 0;
}

