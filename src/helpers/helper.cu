#include "helper.cuh"

#define WINDOW_WIDTH 9
#define WINDOW_HEIGHT 7

ScanLine::ScanLine(int drow, int dcol, bool posdir) {
  this->drow = drow;
  this->dcol = dcol;
  this->posdir = posdir;
}

ScanLines8::ScanLines8() {
  // path8.push_back(ScanLine(1, 0, true));
  // path8.push_back(ScanLine(1, -1, true));
  // path8.push_back(ScanLine(0, -1, false));
  path8.push_back(ScanLine(0, 1, true));
}

__device__ int getDirectionY(size_t path) {
  switch(path) {
    case 0:
      return 1;
    case 1:
      return 1;
    case 2:
      return 1;
    case 3:
      return 0;
    case 4:
      return 0; 
    default:
      return 0;
  }
}

__device__ int getDirectionX(size_t path) {
  switch(path) {
    case 0:
      return 1;
    case 1:
      return 0;
    case 2:
      return -1;
    case 3:
      return -1;
    case 4:
      return 1; 
    default:
      return 0;
  }
}

__device__ int getDirectionPos(size_t path) {
  switch(path) {
    case 0:
      return 1;
    case 1:
      return 1;
    case 2:
      return 1;
    case 3:
      return 1;
    case 4:
      return 0; 
    default:
      return 1;
  }
}

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


uint** toCSCT (short** inArr, size_t rows, size_t cols) {
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
                    sumArr[bitIndex] = (short)((y < 0 || y >= rows || x < 0 || x>=cols ) ? 0 : inArr[y][x]);
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

