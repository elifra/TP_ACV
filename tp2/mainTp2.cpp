
#include <iostream>
 

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>

//#include <opencv2/contrib/contrib.hpp>

using namespace cv;
using namespace std; 

//=======================================================================================
// computeHistogram
//=======================================================================================
void computeHistogram(const Mat& inputComponent, Mat& myHist)
{
	/// Establish the number of bins
	int histSize = 256;
	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 } ;
	const float* histRange = { range };
	bool uniform = true; 
	bool accumulate = false;
	
	/// Compute the histograms:
	calcHist( &inputComponent, 1, 0, Mat(), myHist, 1, &histSize, &histRange, uniform, accumulate );
}

//=======================================================================================
// displayHistogram
//=======================================================================================
void displayHistogram(const Mat& myHist)
{
	// Establish the number of bins
	int histSize = 256;	
	// Draw one histogram
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound( (double) hist_w/histSize );
	Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
	/// Normalize the result to [ 0, histImage.rows ]
	Mat myHistNorm;
	normalize(myHist, myHistNorm, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

	/// Draw for each channel
	for( int i = 1; i < histSize; i++ )
	{
		line( histImage, Point( bin_w*(i-1), hist_h - cvRound(myHistNorm.at<float>(i-1)) ) , Point( bin_w*(i), hist_h - cvRound(myHistNorm.at<float>(i)) ), Scalar( 255, 255, 255), 2, 8, 0 );		
	}
	/// Display
	namedWindow("Display Histo", WINDOW_AUTOSIZE );
	imshow("Display Histo", histImage );
	waitKey();
}

//=======================================================================================
// Mat norm_0_255(InputArray _src)
// Create and return normalized image
//=======================================================================================
Mat norm_0_255(InputArray _src) {
 Mat src = _src.getMat();
 // Create and return normalized image:
 Mat dst;
 switch(src.channels()) {
	case 1:
		normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
		break;
	case 3:
		normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
		break;
	default:
	src.copyTo(dst);
	break;
 }
 return dst;
}

//Conversion BGR --> YCrCb
void bgrToYCbCr(const Mat & src, const Mat & out) {
	cvtColor(src, out, cv::COLOR_BGR2YCrCb);
}

//=======================================================================================
// EQM
//=======================================================================================
double eqm(const Mat & img1, const Mat & img2)
{
	double eqm = 0;
	for(int i=0; i < img1.rows; i++)
	{
		const unsigned char* Mimg1 = img1.ptr<unsigned char>(i);
		const unsigned char* Mimg2 = img2.ptr<unsigned char>(i);
		for(int j=0; j < img1.cols; j++)
		{
			eqm += (double) (abs(Mimg1[j] - Mimg2[j]) * abs(Mimg1[j] - Mimg2[j]));	
		}
	}
	eqm = eqm/(double)(img1.rows * img1.cols);
	return eqm;
}

//=======================================================================================
// psnr
//=======================================================================================
double psnr(const Mat & imgSrc, const Mat & imgDeg)
{
	double EQM = eqm(imgSrc,imgDeg);
	double PSNR = 10*log(255*255/EQM);
	return PSNR;
}

//=======================================================================================
// distortionMap
//=======================================================================================
void distortionMap(const vector<Mat> & imgSrc, const vector<Mat> & imgDeg, Mat &distoMap)
{

	for(int i = 0; i < imgSrc.size(); i++) {
		for(int j = 0; j < imgSrc.size(); j++) {

		}
	}
}

//=======================================================================================
//=======================================================================================
// MAIN
//=======================================================================================
//=======================================================================================
int main(int argc, char** argv){
  if (argc < 2){
    std::cout << "No image data... At least one argument is required! \n";
    return -1;
  }

  Mat inputImageSrc;

  // Ouvrir l'image d'entr�e et v�rifier que l'ouverture du fichier se d�roule normalement
  inputImageSrc = imread(argv[1], IMREAD_COLOR);
  if(!inputImageSrc.data ) { // Check for invalid input
        std::cout <<  "Could not open or find the image " << argv[1] << std::endl ;
		waitKey(0); // Wait for a keystroke in the window
        return -1;
  }
  
  //Save Image BGR
  bool check = imwrite("../Save/imBGR.jpg", inputImageSrc);
  std::cout << check << std::endl;

  //Conversion en YCbCR
  Mat imYCrCb;
  bgrToYCbCr(inputImageSrc,imYCrCb);
  
  //Save ImageYCbCr
  check = imwrite("../Save/imYCrCb.jpg", imYCrCb);
  std::cout << check << std::endl;

  //double computed_eqm = eqm(inputImageSrc, inputImage_compressed);
  //std::cout << "computed_eqm :" << computed_eqm << std::endl;
   
  return 0;
}
