
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
	imwrite("../SaveTP3/im"+name+".bmp", out);
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
// Prédicteur MICD MONO
//=======================================================================================

uchar micdMono(const Mat & imY, int i, int j) {
	if(j > 0) return imY.at<uchar>(i,j-1);
	else return 128;
}

//=======================================================================================
// Prédicteur MICD Bi
//=======================================================================================

uchar micdBi(const Mat & imY, int i, int j) {
	uchar a = 128;
	uchar c = 128;
	if(i > 0) c = imY.at<uchar>(i-1,j);
	if(j > 0) a = imY.at<uchar>(i,j-1);
	return (a+c)/2;
}

//=======================================================================================
// Prédicteur MICDA
//=======================================================================================

uchar micda(const Mat & imY, int i, int j) {
	uchar a = 128;
	uchar b = 128;
	uchar c = 128;
	uchar d = 128;
	if(i > 0) 
	{
		b = imY.at<uchar>(i-1,j-1);
		c = imY.at<uchar>(i-1,j);
		d = imY.at<uchar>(i-1,j+1);
	}
	if(j > 0) a = imY.at<uchar>(i,j-1);
	if(abs(c-b)<abs(a-b)){
		return a;
	}
	else{
		return c;
	}
}

//=======================================================================================
// Quantification
//=======================================================================================

void quantif(int choix,int & val, int & valQuantif) {
	if(choix==0){
		valQuantif = val;
	}
	else{
		if(val>0){
			valQuantif = (int) (val/choix)*choix + choix;
		}
		else{
			valQuantif = (int) (val/choix)*choix;
		}
	}
}


//=======================================================================================
// Quantification Inverse
//=======================================================================================

void quantifInv(int choix,int & val, int & valQuantifInverse) {
	if(choix==0){
		valQuantifInverse = val;
	}
	else{
		valQuantifInverse = (val + val - choix)/2;
	}
}

uchar min(uchar a,uchar b, uchar c,uchar & indice){
	uchar res = a;
	indice = 1;
	if(b<res){
		res = b;
		indice = 2;
	} 
	if(c<res){
		res = c;
		indice = 3;
	}
	return res;
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
  Mat imYUchar;
  saveImage(canaux[0], imYUchar, "Y");

  /*****
   * Prédicteurs
   *****/

  int choixPredicteur;
  cout << "Choisissez un prédicteur(1,2,3)" << endl;
  cout << "1 : MICD Mono-dimensionnelle" << endl;
  cout << "2 : MICD Bi-dimensionnelle" << endl;
  cout << "3 : MICDA" << endl;
  cin >> choixPredicteur;

  int choixQuantif;
  cout << "Choisissez un quantificateur (2,4,6,8,...))" << endl;
  cin >> choixQuantif;
  
  Mat erreurPrediction = Mat(imYUchar.rows, imYUchar.cols, CV_8UC1);
  Mat imageReconstruite = Mat(imYUchar.rows, imYUchar.cols, CV_8UC1);
  Mat canal = Mat(imYUchar.rows, imYUchar.cols, CV_8UC1);
  int prediction = 0;
  int eQuantif = 0;
  int eQuantifInverse = 0;

  for(int i = 0; i < imYUchar.rows; i++) {
	  for(int j = 0; j < imYUchar.cols; j++) {
		  	if(choixPredicteur == 1) prediction = micdMono(imageReconstruite,i,j);
		  	if(choixPredicteur == 2) prediction = micdBi(imageReconstruite,i,j);
		  	if(choixPredicteur == 3) prediction = micda(imageReconstruite,i,j);

		  	erreurPrediction.at<uchar>(i,j) = imYUchar.at<uchar>(i,j)-prediction+128;
			int erreur = (int)(erreurPrediction.at<uchar>(i,j)-128);
			quantif(choixQuantif,erreur, eQuantif);
			canal.at<uchar>(i,j) = eQuantif;
			quantifInv(choixQuantif,eQuantif,eQuantifInverse);
			imageReconstruite.at<uchar>(i,j) = prediction + eQuantifInverse;
	  }
  }
  imwrite("../SaveTP3/imReconstruite.bmp", imageReconstruite);
  
  imwrite("../SaveTP3/imErreursPredictions.bmp", erreurPrediction);

  imwrite("../SaveTP3/imCanal.bmp", canal);
  

  Mat histoErreurs;
  computeHistogram(erreurPrediction, histoErreurs);
  if(choixPredicteur == 1) displayHistogram(histoErreurs, "histoErreursPredictionMICD_MONO");
  if(choixPredicteur == 2) displayHistogram(histoErreurs, "histoErreursPredictionMICD_BI");
  if(choixPredicteur == 3) displayHistogram(histoErreurs, "histoErreursPredictionMICDA");

  /*
  * Image de référence (gris = 128)
  */
  Mat ImageRef = Mat(imYUchar.rows, imYUchar.cols, CV_8UC1);
  for(int i = 0; i < ImageRef.rows; i++) {
	for(int j = 0; j < ImageRef.cols; j++) {
		ImageRef.at<uchar>(i,j) =128;
	}
  }

  /*
  * EQM
  */
  cout<< "EQM de l'erreur :" << eqm(erreurPrediction,ImageRef) <<endl;

  /*
  * PSNR
  */
  cout<< "PSNR de l'erreur :" << psnr(erreurPrediction,ImageRef) <<endl;  

  /*
  * entropie
  */
  cout<< "Entropie de l'erreur :" << entropie(erreurPrediction) <<endl;

  /*****
   * Décodage
   *****/
  Mat decodage  = Mat(canal.rows, canal.cols, CV_8UC1);
  for(int i = 0; i < canal.rows; i++) {
	  for(int j = 0; j < canal.cols; j++) {
		  int indexe = canal.at<uchar>(i,j);
		  quantifInv(choixQuantif,indexe,eQuantifInverse);
		  if(choixPredicteur == 1) prediction = micdMono(decodage,i,j);
		  if(choixPredicteur == 2) prediction = micdBi(decodage,i,j);
	      if(choixPredicteur == 3) prediction = micda(decodage,i,j);

		  decodage.at<uchar>(i,j) = eQuantifInverse + prediction;
	  }
  }

  imwrite("../SaveTP3/imDecode.bmp", decodage);

  /*
  * PSNR
  */
  cout<< "PSNR entre l'image decode et l'image initiale :" << psnr(imYUchar,decodage) <<endl;  
  
  /*
  * entropie
  */
  cout<< "Entropie de l'image decodee :" << entropie(decodage) <<endl;
  cout<< "Entropie de l'image initiale :" << entropie(imYUchar) <<endl;

  cout << "" << endl;
  cout << "PARTIE COMPET" << endl;
  cout << "" << endl;
   
  Mat histoDecode;
  computeHistogram(decodage, histoDecode);
  displayHistogram(histoDecode, "histoImageDecodee");

  
	for(int i = 0; i < histoDecode.rows; i++) {
		if(histoDecode.at<float>(i) > 0) std::cout<<i<<std::endl;
	}

   /** PARTIE COMPETITION **/
  uchar prediction1 =0;
  uchar prediction2 =0;
  uchar prediction3 =0;
  uchar erreurPrediction1 =0;
  uchar erreurPrediction2 =0;
  uchar erreurPrediction3 =0; 
  Mat compet = Mat(imYUchar.rows, imYUchar.cols, CV_8UC1);
  erreurPrediction  = Mat(canal.rows, canal.cols, CV_8UC1);
  canal  = Mat(canal.rows, canal.cols, CV_8UC1);
  imageReconstruite  = Mat(canal.rows, canal.cols, CV_8UC1);
  uchar indice = 1;
  for(int i = 0; i < imYUchar.rows; i++) {
	  for(int j = 0; j < imYUchar.cols; j++) {
		  	prediction1 = micdMono(imageReconstruite,i,j);
		  	prediction2 = micdBi(imageReconstruite,i,j);
		  	prediction3 = micda(imageReconstruite,i,j);

		  	erreurPrediction1 = imYUchar.at<uchar>(i,j)-prediction1+128;
		  	erreurPrediction2 = imYUchar.at<uchar>(i,j)-prediction2+128;
		  	erreurPrediction3 = imYUchar.at<uchar>(i,j)-prediction3+128;
			erreurPrediction.at<uchar>(i,j) = min(erreurPrediction1,erreurPrediction2,erreurPrediction3,indice);
			compet.at<uchar>(i,j) = indice;
			int erreur = (int)(erreurPrediction.at<uchar>(i,j)-128);
			quantif(choixQuantif,erreur, eQuantif);
			canal.at<uchar>(i,j) = eQuantif;
			quantifInv(choixQuantif,eQuantif,eQuantifInverse);
			if(indice ==1) prediction = prediction1;
			if(indice ==2) prediction = prediction2;
			if(indice ==3) prediction = prediction3;
			imageReconstruite.at<uchar>(i,j) = prediction + eQuantifInverse;
	  }
  }
  imwrite("../SaveTP3/imReconstruiteCompet.bmp", imageReconstruite);
  
  imwrite("../SaveTP3/imErreursPredictionsCompet.bmp", erreurPrediction);

  imwrite("../SaveTP3/imCanalCompet.bmp", canal);

  decodage  = Mat(canal.rows, canal.cols, CV_8UC1);
  int cpt1 = 0;
  int cpt2 = 0;
  int cpt3 = 0;
  for(int i = 0; i < canal.rows; i++) {
	  for(int j = 0; j < canal.cols; j++) {
		  int indexe = canal.at<uchar>(i,j);
		  quantifInv(choixQuantif,indexe,eQuantifInverse);
		  if(compet.at<uchar>(i,j) == 1) {
			  cpt1++;
			  prediction = micdMono(decodage,i,j);
		  }
		  if(compet.at<uchar>(i,j) == 2){
			  cpt2++;
			  prediction = micdBi(imageReconstruite,i,j);
		  }
	      if(compet.at<uchar>(i,j) == 3){
			  cpt3++;
			  prediction = micda(imageReconstruite,i,j);
		  } 

		  decodage.at<uchar>(i,j) = eQuantifInverse + prediction;
	  }
  }

  imwrite("../SaveTP3/imDecodeCompet.bmp", decodage);
  
  /*
  * PSNR
  */
  cout<< "PSNR entre l'image decode et l'image initiale :" << psnr(imYUchar,decodage) <<endl;  
  
  /*
  * entropie
  */
  cout<< "Entropie de l'image decodee :" << entropie(decodage) <<endl;
  cout<< "Entropie de l'image initiale :" << entropie(imYUchar) <<endl;

  /*
  Analyse
  */
  cout<< "Taux prédicteur 1 : " << (double)(cpt1)/(double)(canal.rows*canal.cols) <<endl;
  cout<< "Taux prédicteur 2 : " << (double)(cpt2)/(double)(canal.rows*canal.cols) <<endl;
  cout<< "Taux prédicteur 3 : " << (double)(cpt3)/(double)(canal.rows*canal.cols) <<endl;
  return 0;
}
