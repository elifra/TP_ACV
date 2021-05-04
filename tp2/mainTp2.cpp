
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>

//#include <opencv2/contrib/contrib.hpp>

using namespace cv;
using namespace std; 

void saveImage(const Mat & src, Mat & out, std::string name) {
	Mat tmp;
	src.convertTo(tmp, CV_8UC1, 255);
	out = tmp;
	imwrite("../SaveTP2/im"+name+".bmp", out);
}

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
void displayHistogram(const Mat& myHist, std::string name)
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
	Mat histImageSave;
	saveImage(histImage, histImageSave, "histo"+name);
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
void bgrToYCrCb(const Mat & src, Mat & out) {
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
	double PSNR = 10*log10(255*255/EQM);
	return PSNR;
}

double psnrFloat(const Mat & imgSrc, const Mat & imgDeg)
{
	//Conversion
	Mat imgSrcUCHAR;
	imgSrc.convertTo(imgSrcUCHAR, CV_8UC1, 255);
	Mat imgDegUCHAR;
	imgDeg.convertTo(imgDegUCHAR, CV_8UC1, 255);

	//Calculs
	double EQM = eqm(imgSrcUCHAR,imgDegUCHAR);
	double PSNR = 10*log10(255*255/EQM);
	return PSNR;
}

//=======================================================================================
// entropie
//=======================================================================================
double entropie(Mat & imgSrc)
{
	Mat histo;
   	computeHistogram(imgSrc,histo);
	double res = 0;
	for(int i = 0; i < histo.rows; i++) {
		float proba = histo.at<float>(i)/(imgSrc.rows*imgSrc.cols);
		if(proba > 0) res += proba*log2(proba);
	}
	return -res;
}

//=======================================================================================
// Coeffs dct
//=======================================================================================
void dctCoeffs(const Mat & in, Mat & outColor, Mat & out) {
	Mat inUCHAR;
	in.convertTo(inUCHAR, CV_8UC1, 255);
	Mat outCalcul = Mat(inUCHAR.rows, inUCHAR.cols, CV_8UC1);
	out = Mat(inUCHAR.rows, inUCHAR.cols, CV_8UC1);

	double min, max;
	minMaxLoc(inUCHAR,&min,&max);
	for(int i = 0; i < in.rows; i++) {
		for(int j = 0; j < in.cols; j++) {
			out.at<uchar>(i,j) = log(1+abs(inUCHAR.at<uchar>(i,j)))/log(1+max)*255;
		}
	}
	applyColorMap(out, outColor, COLORMAP_JET);
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
  Mat inputImageSrcTmp = imread(argv[1], IMREAD_COLOR);
  if(!inputImageSrcTmp.data ) { // Check for invalid input
        std::cout <<  "Could not open or find the image " << argv[1] << std::endl ;
		waitKey(0); // Wait for a keystroke in the window
        return -1;
  }

  inputImageSrcTmp.convertTo(inputImageSrc, CV_32FC3);
  inputImageSrc=inputImageSrc/255;

  //Conversion en YCbCR
  Mat imYCrCb;
  bgrToYCrCb(inputImageSrc,imYCrCb);

  //Save ImageYCbCr
  Mat imSaveYCrCb;
  saveImage(imYCrCb, imSaveYCrCb, "YCbCr");	
  
  //std::cout << check << std::endl;
  
  std::vector<Mat> canaux;
  split(imYCrCb,canaux);
  std::string noms[] = {"Y","Cr","Cb"};
  int i = 0;
  for(Mat im : canaux) {
	Mat imSaveCanaux;
	saveImage(im, imSaveCanaux, noms[i]);
	i++;
  }

  /*****
   * DCT
   *****/
  
  i = 0;
  Mat dIm;
  Mat dinvIm;
  int choixFiltre;
  cout << "Choisissez un filtre (1,2,3,4,5,6)" << endl;
  cout << "0 : pas de filtre" << endl;
  cout << "1 : carré en bas à droite" << endl;
  cout << "2 : rectangle horizontal en bas" << endl;
  cout << "3 : rectangle vertical à droite" << endl;
  cout << "4 : carré en haut à droite" << endl;
  cout << "5 : carré en haut à gauche" << endl;
  cout << "6 : carré en bas à gauche" << endl;
  cin >> choixFiltre;

  for(Mat im : canaux) {
	//dct
	dct(im,dIm);
	Mat imSaveDct;
	saveImage(dIm, imSaveDct, "DCT"+noms[i]);

	//Filtre
	if(choixFiltre == 1) dIm(Rect(dIm.rows/2, dIm.cols/2, dIm.rows/2, dIm.cols/2)) = 0;
	else if(choixFiltre == 2) dIm(Rect(0, dIm.cols/2, dIm.rows, dIm.cols/2)) = 0;
	else if(choixFiltre == 3) dIm(Rect(dIm.rows/2, 0, dIm.rows/2, dIm.cols)) = 0;
	else if(choixFiltre == 4) dIm(Rect(dIm.rows/2, 0, dIm.rows/2, dIm.cols/2)) = 0;
	else if(choixFiltre == 5) dIm(Rect(0, 0, dIm.rows/2, dIm.cols/2)) = 0;
	else if(choixFiltre == 6) dIm(Rect(0, dIm.cols/2, dIm.rows/2, dIm.cols/2)) = 0;

	//dct coeffs
	Mat coeffsDCTColor;
	Mat coeffsDCT;
	dctCoeffs(dIm,coeffsDCTColor,coeffsDCT);
	Mat imSaveCoeffsDctColor;
	saveImage(coeffsDCTColor, imSaveCoeffsDctColor, "CoeffsDCTColor"+noms[i]);

	Mat imSaveCoeffsDcttest;
	saveImage(coeffsDCT, imSaveCoeffsDcttest, "CoeffsDCT"+noms[i]);

	//histo dct coeffs
	Mat histoCanaux;
	Mat imUCHAR;
	im.convertTo(imUCHAR, CV_8UC1, 255);
	computeHistogram(imUCHAR, histoCanaux);
	displayHistogram(histoCanaux, noms[i]);

	Mat histoCoeffsDCT;
	computeHistogram(coeffsDCT, histoCoeffsDCT);
	displayHistogram(histoCoeffsDCT, noms[i]+"CoeffDCT");

	//Entropie
	cout << "Entropie " << noms[i] << " : " << entropie(imUCHAR) << endl;
	cout << "Entropie " << noms[i] << "CoeffsDCT : " << entropie(coeffsDCT) << endl;

	//dct inv
	dct(dIm,dinvIm, DCT_INVERSE);
	Mat imSaveInvDct;
	saveImage(dinvIm, imSaveInvDct, "DCTInv"+noms[i]);

	std::cout << "PSNR " << noms[i] << " = " << psnrFloat(im, dinvIm) << std::endl;
	i++;
  }

  
   
  return 0;
}
