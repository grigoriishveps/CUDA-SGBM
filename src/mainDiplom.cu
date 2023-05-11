#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/ximgproc.hpp"
#include <iostream>
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/core/utility.hpp"

using namespace cv;
using namespace cv::ximgproc;
using namespace cv::cuda;
using namespace std;

Rect computeROI(Size2i src_sz, Ptr<StereoMatcher> matcher_instance)
{
    int min_disparity = matcher_instance->getMinDisparity();
    int num_disparities = matcher_instance->getNumDisparities();
    int block_size = matcher_instance->getBlockSize();

    int bs2 = block_size/2;
    int minD = min_disparity, maxD = min_disparity + num_disparities - 1;

    int xmin = maxD + bs2;
    int xmax = src_sz.width + minD - bs2;
    int ymin = bs2;
    int ymax = src_sz.height - bs2;

    Rect r(xmin, ymin, xmax - xmin, ymax - ymin);
    return r;
}

int main( int argc, char** argv )
{
    Mat left_for_matcher, right_for_matcher;

    double solving_time = 0;    
    Mat left = cv::imread("./src/images/warLeft.jpg",cv::IMREAD_COLOR);
    Mat right = cv::imread("./src/images/warRight.jpg",cv::IMREAD_COLOR);
    

    left_for_matcher  = left.clone();
    right_for_matcher = right.clone();

    int wsize = 3;
    int max_disp = 64;
    Mat left_disp;
    Ptr<DisparityWLSFilter> wls_filter;

    cout << "START"<<endl;
    double matching_time = (double)getTickCount();

    Ptr<StereoSGBM> matcher  = StereoSGBM::create(0,max_disp,wsize);
    matcher->setUniquenessRatio(0);
    matcher->setDisp12MaxDiff(1000000);
    matcher->setSpeckleWindowSize(0);
    matcher->setP1(24*wsize*wsize);
    matcher->setP2(96*wsize*wsize);
    matcher->setMode(StereoSGBM::MODE_HH);

    computeROI(left_for_matcher.size(),matcher);

    wls_filter = createDisparityWLSFilterGeneric(false);
    wls_filter->setDepthDiscontinuityRadius((int)ceil(0.5*wsize));

    matcher->compute(left_for_matcher,right_for_matcher,left_disp);
    matching_time = ((double)getTickCount() - matching_time)/getTickFrequency();
    
    cout.precision(5);
    cout << "READY: "<< matching_time  << endl;
    
    double vis_mult = 1.0;
    Mat raw_disp_vis;
    getDisparityVis(left_disp,raw_disp_vis,vis_mult);
    namedWindow("raw disparity", WINDOW_AUTOSIZE);
    imshow("raw disparity", raw_disp_vis);
    
    // no off program
    while(1)
    {
        char key = (char)waitKey();
        if( key == 27 || key == 'q' || key == 'Q') // 'ESC'
            break;
    }

    return 0;
}