
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
	bool check = imwrite("../SaveTP1/histo"+name+".jpg", histImage);
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
void bgrToYCbCr(Mat & src, Mat & out) {
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

  // Ouvrir l'image d'entr???e et v???rifier que l'ouverture du fichier se d???roule normalement
  inputImageSrc = imread(argv[1], IMREAD_COLOR);
  if(!inputImageSrc.data ) { // Check for invalid input
        std::cout <<  "Could not open or find the image " << argv[1] << std::endl ;
		waitKey(0); // Wait for a keystroke in the window
        return -1;
  }

  Mat inputImageDeg2;

  // Ouvrir l'image d'entr???e et v???rifier que l'ouverture du fichier se d???roule normalement
  inputImageDeg2 = imread(argv[2], IMREAD_COLOR);
  if(!inputImageDeg2.data ) { // Check for invalid input
        std::cout <<  "Could not open or find the image " << argv[1] << std::endl ;
		waitKey(0); // Wait for a keystroke in the window
        return -1;
  }

  Mat inputImageDeg10;

  // Ouvrir l'image d'entr???e et v???rifier que l'ouverture du fichier se d???roule normalement
  inputImageDeg10 = imread(argv[3], IMREAD_COLOR);
  if(!inputImageDeg10.data ) { // Check for invalid input
        std::cout <<  "Could not open or find the image " << argv[1] << std::endl ;
		waitKey(0); // Wait for a keystroke in the window
        return -1;
  }

  Mat inputImageDeg80;

  // Ouvrir l'image d'entr???e et v???rifier que l'ouverture du fichier se d???roule normalement
  inputImageDeg80 = imread(argv[4], IMREAD_COLOR);
  if(!inputImageDeg80.data ) { // Check for invalid input
        std::cout <<  "Could not open or find the image " << argv[1] << std::endl ;
		waitKey(0); // Wait for a keystroke in the window
        return -1;
  }
  
  //Save Image BGR
  bool check = imwrite("../SaveTP1/imBGR.jpg", inputImageSrc);
  //std::cout << check << std::endl;

  //Conversion en YCbCR
  Mat imYCrCb;
  bgrToYCbCr(inputImageSrc,imYCrCb);
  
  //Save ImageYCbCr
  check = imwrite("../SaveTP1/imYCrCb.jpg", imYCrCb);
  //std::cout << check << std::endl;

  std::vector<Mat> canaux;
  split(imYCrCb,canaux);
  std::string noms[] = {"Y","Cr","Cb"};
  int i = 0;
  for(Mat im : canaux) {
	check = imwrite("../SaveTP1/im"+noms[i]+".jpg", im);
	i++;
  }

  /*****
   * PSNR
   *****/
  Mat imYCrCbDEG2;
  bgrToYCbCr(inputImageDeg2,imYCrCbDEG2);
  check = imwrite("../SaveTP1/imYCrCbDEG2.jpg", imYCrCbDEG2);
  std::vector<Mat> canauxDEG;
  split(imYCrCbDEG2,canauxDEG);
  i = 0;
  for(Mat im : canauxDEG) {
	check = imwrite("../SaveTP1/im"+noms[i]+"DEG2.jpg", im);
	i++;
  }
  double PSNR;
  for(int j = 0; j < 3; j++) {
	PSNR = psnr(canaux[j],canauxDEG[j]);
	cout << "PSNR pour canal : " << noms[j] << endl;
	cout << PSNR << endl;
  }

   /*****
   * Carte erreur
   *****/
  Mat Y = canaux[0];
  Mat Ydeg = canauxDEG[0];
  Mat carteErreur = Y - Ydeg + 128;
  check = imwrite("../SaveTP1/carteErreurDeg2.jpg", carteErreur);

  /*****
   * 3.4
   *****/
  Mat imYCbCrDeg10;
  bgrToYCbCr(inputImageDeg10,imYCbCrDeg10);
  Mat imYCbCrDeg80;
  bgrToYCbCr(inputImageDeg80,imYCbCrDeg80);

  std::vector<Mat> canauxDeg10;
  split(imYCbCrDeg10, canauxDeg10);
  std::vector<Mat> canauxDeg80;
  split(imYCbCrDeg80, canauxDeg80);

  i = 0;
  for(Mat im : canauxDeg10) {
	check = imwrite("../SaveTP1/im"+noms[i]+"DEG10.jpg", im);
	i++;
  }
  i = 0;
  for(Mat im : canauxDeg80) {
	check = imwrite("../SaveTP1/im"+noms[i]+"DEG80.jpg", im);
	i++;
  }
  PSNR = psnr(canaux[0],canauxDeg10[0]);
  cout << "PSNR pour canal Y image de base et image Deg10 " << endl;
  cout << PSNR << endl;

  PSNR = psnr(canaux[0],canauxDeg80[0]);
  cout << "PSNR pour canal Y image de base et image Deg80 " << endl;
  cout << PSNR << endl;

  /*****
   * Courbe
   *****/
  PSNR = psnr(imYCrCb,imYCrCbDEG2);
  cout << "PSNR image de base et image Deg2 " << endl;
  cout << PSNR << endl;

  PSNR = psnr(imYCrCb,imYCbCrDeg10);
  cout << "PSNR image de base et image Deg10 " << endl;
  cout << PSNR << endl;

  PSNR = psnr(imYCrCb,imYCbCrDeg80);
  cout << "PSNR image de base et image Deg80 " << endl;
  cout << PSNR << endl;

  /*****
   * Entropie
   *****/
   cout << "Entropie Y Image de base" << endl;
   cout << entropie(canaux[0]) << endl;
   cout << "Entropie Y Deg2" << endl;
   cout << entropie(canauxDEG[0]) << endl;
   cout << "Entropie Y Deg10" << endl;
   cout << entropie(canauxDeg10[0]) << endl;
   cout << "Entropie Y Deg80" << endl;
   cout << entropie(canauxDeg80[0]) << endl;

  //double computed_eqm = eqm(inputImageSrc, inputImage_compressed);
  //std::cout << "computed_eqm :" << computed_eqm << std::endl;

   /*****
   * Histogramme des cartes d'erreur
   *****/

  //Image de base
  Mat carteErreurImagedeBase = Y - Y + 128;
  check = imwrite("../SaveTP1/carteErreurImagedebase.jpg", carteErreurImagedeBase);

  cout << "Entropie Y carte erreur Image de base" << endl;
  cout << entropie(carteErreurImagedeBase) << endl;

  Mat histoCarteErreurBase;
  computeHistogram(carteErreurImagedeBase,histoCarteErreurBase);
  displayHistogram(histoCarteErreurBase, "CarteErreurBase");

  //Image degQ2
  cout << "Entropie Y carte erreur Image Deg2" << endl;
  cout << entropie(carteErreur) << endl;

  Mat histoCarteErreurDeg2;
  computeHistogram(carteErreur,histoCarteErreurDeg2);
  displayHistogram(histoCarteErreurDeg2, "CarteErreurDeg2");

  //Image degQ10
  Mat Ydeg10 = canauxDeg10[0];
  Mat carteErreurImageDeg10 = Y - Ydeg10 + 128;
  check = imwrite("../SaveTP1/carteErreurImageDeg10.jpg", carteErreurImageDeg10);

  cout << "Entropie Y carte erreur Image Deg10" << endl;
  cout << entropie(carteErreurImageDeg10) << endl;

  Mat histoCarteErreurDeg10;
  computeHistogram(carteErreurImageDeg10,histoCarteErreurDeg10);
  displayHistogram(histoCarteErreurDeg10, "CarteErreurDeg10");

  //Image degQ80
  Mat Ydeg80 = canauxDeg80[0];
  Mat carteErreurImageDeg80 = Y - Ydeg80 + 128;
  check = imwrite("../SaveTP1/carteErreurImageDeg80.jpg", carteErreurImageDeg80);

  cout << "Entropie Y carte erreur Image Deg80" << endl;
  cout << entropie(carteErreurImageDeg80) << endl;

  Mat histoCarteErreurDeg80;
  computeHistogram(carteErreurImageDeg80,histoCarteErreurDeg80);
  displayHistogram(histoCarteErreurDeg80, "CarteErreurDeg80");

   
  return 0;
}
