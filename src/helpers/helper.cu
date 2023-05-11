#include "helper.cuh"

#define WINDOW_WIDTH 9
#define WINDOW_HEIGHT 7

__global__ void censusTransform(unsigned char *img, unsigned char *census, size_t cols) {
  int row = blockIdx.x + 1;  // block index
  int col = blockIdx.y + 1;

  unsigned char val = 0;
  for (int drow = -1; drow <= 1; drow += 2) {
    for (int dcol = -1; dcol <= 1; dcol += 2) {
      unsigned char tmp = img[cols * (row + drow) + col + dcol];
      val = (val + (tmp < img[cols * row + col] ? 0 : 1)) << 1;        
    }
  }

  census[cols*row + col] = (unsigned char) val; 
}


uint** toCSCT (short* inArr, size_t rows, size_t cols) {
    uint **resArr = new uint*[rows];

    for (int i = 0; i < rows; ++i){
        resArr[i] = new uint[cols];
    }

    short* sumArr = new short[WINDOW_HEIGHT * WINDOW_WIDTH];

    for (int i = 0; i < rows; ++i){
        for (int j = 0; j < cols; ++j) {
            // TODO change on dynamic
            uint sum = 0x00000000;
            
            for (int y = i - int(WINDOW_HEIGHT / 2), bitIndex = 0; y < i + int(WINDOW_HEIGHT / 2) ; ++y){
                for (int x = j - int(WINDOW_WIDTH / 2); x < j + int(WINDOW_WIDTH / 2); ++x, bitIndex++) {
                    sumArr[bitIndex] = (short)((y < 0 || y >= rows || x < 0 || x>=cols ) ? 0 : inArr[y * cols + x]);
                }
            } 

            for (int x = 0, y = WINDOW_HEIGHT * WINDOW_WIDTH - 1; x != y; x++,y--) {
              sum = (sum << 1) + (sumArr[x] > sumArr[y] ? 1 : 0);
            }
            
            resArr[i][j] = sum;
        }   
    }    

    delete[] sumArr;  

    return resArr;
}

short** toMatArray (cv::Mat mat) {
    short **array = new short*[mat.rows];

    for (int i=0; i<mat.rows; ++i){
        array[i] = new short[mat.cols];
        for (int j=0; j<mat.cols; ++j){
          array[i][j] = mat.at<short>(i, j);
        }
    }

    return array;
}

