
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
// recopier Block
//=======================================================================================
void RecopieBlock(const Mat & in, Mat & out,int i,int j) {
	for(int i2 = 0; i2 < 8; i2++) {
		for(int j2 = 0; j2 < 8; j2++) {
			out.at<float>(i+i2,j+j2) =in.at<float>(i2,j2);
		}
	}
}

//=======================================================================================
// filtre A
//=======================================================================================
void filtreA(Mat & in) {
	Mat filtre = Mat(8,8, CV_32FC1);
	for(int i=0;i<in.rows;i++){
		for(int j=0;j<in.cols;j++){
			filtre.at<float>(i,j) =0;
		}
	}

	filtre.at<float>(0,0) =1.f;
	filtre.at<float>(0,1) =1.f;
	filtre.at<float>(0,2) =1.f;
	filtre.at<float>(0,3) =1.f;
	filtre.at<float>(0,4) =1.f;
	filtre.at<float>(1,0) =1.f;
	filtre.at<float>(1,1) =1.f;
	filtre.at<float>(1,2) =1.f;
	filtre.at<float>(1,3) =1.f;
	filtre.at<float>(2,0) =1.f;
	filtre.at<float>(2,1) =1.f;
	filtre.at<float>(2,2) =1.f;
	filtre.at<float>(3,0) =1.f;
	filtre.at<float>(3,1) =1.f;
	filtre.at<float>(4,0) =1.f;
	for(int i=0;i<in.rows;i++){
		for(int j=0;j<in.cols;j++){
			in.at<float>(i,j) =in.at<float>(i,j)*filtre.at<float>(i,j);
		}
	}
}

//=======================================================================================
// filtre B
//=======================================================================================
void filtreB(Mat & in,int mode) {
	Mat filtre = Mat(8,8, CV_32FC1);

	filtre.at<float>(0,0) =16.f/255.f;
	filtre.at<float>(0,1) =11.f/255.f;
	filtre.at<float>(0,2) =10.f/255.f;
	filtre.at<float>(0,3) =16.f/255.f;
	filtre.at<float>(0,4) =24.f/255.f;
	filtre.at<float>(0,5) =40.f/255.f;
	filtre.at<float>(0,6) =51.f/255.f;
	filtre.at<float>(0,7) =61.f/255.f;

	filtre.at<float>(1,0) =12.f/255.f;
	filtre.at<float>(1,1) =12.f/255.f;
	filtre.at<float>(1,2) =14.f/255.f;
	filtre.at<float>(1,3) =19.f/255.f;
	filtre.at<float>(1,4) =26.f/255.f;
	filtre.at<float>(1,5) =58.f/255.f;
	filtre.at<float>(1,6) =60.f/255.f;
	filtre.at<float>(1,7) =55.f/255.f;

	filtre.at<float>(2,0) =14.f/255.f;
	filtre.at<float>(2,1) =13.f/255.f;
	filtre.at<float>(2,2) =16.f/255.f;
	filtre.at<float>(2,3) =24.f/255.f;
	filtre.at<float>(2,4) =40.f/255.f;
	filtre.at<float>(2,5) =57.f/255.f;
	filtre.at<float>(2,6) =69.f/255.f;
	filtre.at<float>(2,7) =26.f/255.f;

	filtre.at<float>(3,0) =14.f/255.f;
	filtre.at<float>(3,1) =17.f/255.f;
	filtre.at<float>(3,2) =22.f/255.f;
	filtre.at<float>(3,3) =29.f/255.f;
	filtre.at<float>(3,4) =51.f/255.f;
	filtre.at<float>(3,5) =87.f/255.f;
	filtre.at<float>(3,6) =80.f/255.f;
	filtre.at<float>(3,7) =62.f/255.f;

	filtre.at<float>(4,0) =18.f/255.f;
	filtre.at<float>(4,1) =22.f/255.f;
	filtre.at<float>(4,2) =37.f/255.f;
	filtre.at<float>(4,3) =56.f/255.f;
	filtre.at<float>(4,4) =68.f/255.f;
	filtre.at<float>(4,5) =109.f/255.f;
	filtre.at<float>(4,6) =103.f/255.f;
	filtre.at<float>(4,7) =77.f/255.f;

	filtre.at<float>(5,0) =24.f/255.f;
	filtre.at<float>(5,1) =35.f/255.f;
	filtre.at<float>(5,2) =55.f/255.f;
	filtre.at<float>(5,3) =64.f/255.f;
	filtre.at<float>(5,4) =81.f/255.f;
	filtre.at<float>(5,5) =104.f/255.f;
	filtre.at<float>(5,6) =113.f/255.f;
	filtre.at<float>(5,7) =92.f/255.f;

	filtre.at<float>(6,0) =49.f/255.f;
	filtre.at<float>(6,1) =64.f/255.f;
	filtre.at<float>(6,2) =78.f/255.f;
	filtre.at<float>(6,3) =87.f/255.f;
	filtre.at<float>(6,4) =103.f/255.f;
	filtre.at<float>(6,5) =121.f/255.f;
	filtre.at<float>(6,6) =120.f/255.f;
	filtre.at<float>(6,7) =101.f/255.f;
	
	filtre.at<float>(7,0) =72.f/255.f;
	filtre.at<float>(7,1) =92.f/255.f;
	filtre.at<float>(7,2) =95.f/255.f;
	filtre.at<float>(7,3) =98.f/255.f;
	filtre.at<float>(7,4) =112.f/255.f;
	filtre.at<float>(7,5) =100.f/255.f;
	filtre.at<float>(7,6) =103.f/255.f;
	filtre.at<float>(7,7) =99.f/255.f;

	if(mode == 1){
		for(int i=0;i<in.rows;i++){
			for(int j=0;j<in.cols;j++){
				in.at<float>(i,j) = round( in.at<float>(i,j)/filtre.at<float>(i,j) );
			}
		}
	}
	if(mode == 2){
		for(int i=0;i<in.rows;i++){
			for(int j=0;j<in.cols;j++){
				in.at<float>(i,j) = in.at<float>(i,j)*filtre.at<float>(i,j);
			}
		}
	}
}

//=======================================================================================
// DCT par blocs
//=======================================================================================
void dctParBlocs(const Mat & in, Mat & DCT,Mat & DCTinv, int choix) {
	for(int i = 0; i < in.rows; i = i+8) {
		for(int j = 0; j < in.cols; j = j+8) {
			Mat bloc = in(Rect(j,i,8,8));

			Mat DCTBloc;
			dct(bloc,DCTBloc);
			if(choix == 1) filtreA(DCTBloc);
			if(choix == 2) filtreB(DCTBloc,1);
			RecopieBlock(DCTBloc,DCT,i,j);

			Mat DCTBlocInv;
			if(choix == 2) filtreB(DCTBloc,2);
			dct(DCTBloc,DCTBlocInv,DCT_INVERSE);
			RecopieBlock(DCTBlocInv,DCTinv,i,j);
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
  cout << "7 : annulation fréquence nulle" << endl;
  cin >> choixFiltre;

  /*
  * Étude DCT image 
  */

  for(Mat im : canaux) {
	//dct
	dct(im,dIm);
	Mat imSaveDct;
	saveImage(dIm, imSaveDct, "DCT"+noms[i]);

	//Filtre
	if(choixFiltre == 1) dIm(Rect(dIm.cols/2, dIm.rows/2, dIm.cols/2, dIm.rows/2)) = 0;
	else if(choixFiltre == 2) dIm(Rect(0, dIm.rows/2, dIm.cols, dIm.rows/2)) = 0;
	else if(choixFiltre == 3) dIm(Rect(dIm.cols/2, 0, dIm.cols/2, dIm.rows)) = 0;
	else if(choixFiltre == 4) dIm(Rect(dIm.cols/2, 0, dIm.cols/2, dIm.rows/2)) = 0;
	else if(choixFiltre == 5) dIm(Rect(0, 0, dIm.cols/2, dIm.rows/2)) = 0;
	else if(choixFiltre == 6) dIm(Rect(0, dIm.rows/2, dIm.cols/2, dIm.rows/2)) = 0;
	if (choixFiltre == 7) dIm.at<float>(0, 0) = 0;

	//Enregistrement filtre
	Mat masque = Mat(dIm.rows,dIm.cols, CV_32FC1);
	for (int ligne = 0; ligne < dIm.rows; ligne++) {
		for (int colonne = 0; colonne < dIm.cols; colonne++) {
			masque.at<float>(ligne, colonne) = 1;
		}
	}
	if (choixFiltre == 1) masque(Rect(dIm.cols / 2, dIm.rows / 2, dIm.cols / 2, dIm.rows / 2)) = 0;
	else if (choixFiltre == 2) masque(Rect(0, dIm.rows / 2, dIm.cols, dIm.rows / 2)) = 0;
	else if (choixFiltre == 3) masque(Rect(dIm.cols / 2, 0, dIm.cols / 2, dIm.rows)) = 0;
	else if (choixFiltre == 4) masque(Rect(dIm.cols / 2, 0, dIm.cols / 2, dIm.rows / 2)) = 0;
	else if (choixFiltre == 5) masque(Rect(0, 0, dIm.cols / 2, dIm.rows / 2)) = 0;
	else if (choixFiltre == 6) masque(Rect(0, dIm.rows / 2, dIm.cols / 2, dIm.rows / 2)) = 0;

	Mat saveMasque;
	saveImage(masque, saveMasque, "Masque");

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

  /*
  * Étude DCT blocs 8*8
  */
   cout << "Nombre de lignes : " << imYCrCb.rows << endl;
   cout << "Nombre de colonnes : " << imYCrCb.cols << endl;

  i = 0;
  cout << "DCT par blocs : " << endl;
  cout << "0 : sans filtre " << endl;
  cout << "1 : avec filtre a) : " << endl;
  cout << "2 : avec filtre b) : " << endl;
  int choix;
  cin >> choix;
  for(Mat im : canaux) {
	   	Mat DCT = Mat(im.rows,im.cols, CV_32FC1);
	  	Mat DCTinv = Mat(im.rows,im.cols, CV_32FC1);
	  	dctParBlocs(im,DCT,DCTinv, choix);
	  	Mat imSaveDct;
		saveImage(DCT, imSaveDct, "DCTBlocs"+noms[i]);
		Mat imSaveInvDct;
		saveImage(DCTinv, imSaveInvDct, "DCTBlocsInv"+noms[i]);
		i++;
  }


  
   
  return 0;
}
