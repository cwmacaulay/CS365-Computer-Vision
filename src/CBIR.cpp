/* Charles Macaulay
   02-23-2016
   CS 365
   
   CBIR.cpp

   Some code modeled on getDirectoryListing.cpp by Bruce Maxwell
*/

#include <cstdio>
#include <cstdlib>
#include <dirent.h>
#include <stack>
#include <sys/stat.h>
#include <cmath>
#include <string>
#include <vector>
#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace cv; 
using namespace std;


// function declarations
void processDirectory( char* directory );
void processFile( struct dirent* file );
void processEntity( struct dirent* entity );
vector<string> task1_query( Mat query, int results );
vector<string> task2_query( Mat query, int results );
vector<string> task3_query( Mat query, int results );
vector<string> task4_query( Mat query, int results );
vector<string> task5_query( Mat query, int results );
void displayQandR( vector<string> results );








// this keeps track of the path to the directory throughout the process. 
string path;
// because need to change path name different if just entering imageDB
int depth;
// so we can cap off how many images we read into FV file for time's sake. 
int counter;
// specifies which feature vectors need to go into the feature vectors file
int comptype;
// files to write the FV's to:
ofstream myfile1;
// specifies how many top result images the query functions need to return
int top_length;










/* main function takes in user input and operates program accordingly. */
int main(int argc, char *argv[]) {
	char dirname[256];
	DIR *dirp;
	// temp dirent
	struct dirent *dp;

	
	// by default, look at the current directory
	strcpy(dirname, ".");

	// make sure the user provided a directory path and taskFV
	if(argc > 1) {
		strcpy(dirname, argv[1]);
		
	} else {
	  printf( "please provide an image database!\n");
	  exit(-1);
	}

	// Prompt user for type of comparison.
	printf("\n\n\n\n\n\nEnter comparison type:\n[1]\t\tbaseline matching\n[2]\t\tbaseline histogram matching\n[3]\t\tmultiple histogram matching\n[4]\t\ttexture and color\n[5]\t\tcustom\n");
	cin >> comptype;
	if (( comptype < 1 ) || ( comptype > 5 )){
	  printf("Error: the input %d is not a valid comparison type.\n", comptype );
	  exit(-1);
	}
	
	// Prompt user for whether program needs to initialize feature vectors or run a query image.
	int runmode;
	printf("\n\n\n\n\n\nEnter run mode:\n[1]\t\tinitialize feature vectors\n[2]\t\trun a query image\n");
	cin >> runmode;
	cin.ignore();
	if (( runmode != 1 ) && ( runmode != 2)){
	  printf("Error: the input %d is not a valid run mode.\n", runmode );
	  exit(-1);
	  
	} else if (runmode == 1){
	  printf("\n\nRunmode: Initialize Feature Vectors\n\n\n");
	  printf("Accessing directory %s\n\n", dirname);
	  path = dirname;
	  depth = 0;
	  counter = 0;

	  // create the appropriate feature vectors file.
	  if ( comptype == 1 ){
	    myfile1.open ("task1_FV.txt");
	  } else if ( comptype == 2 ){
	    myfile1.open ("task2_FV.txt");
	  } else if ( comptype == 3 ){
	    myfile1.open ("task3_FV.txt");
	  } else if ( comptype == 4 ) {
	    myfile1.open ("task4_FV.txt");
	  } else if ( comptype == 5 ){
	    myfile1.open ("task5_FV.txt");
	  }
	  
	  processDirectory( argv[1] );
	  // close the feature vectors file. This is after folder has been gone through.
	  myfile1.close();
	
	  return(0);
	} else if ( runmode == 2 ){
	  // Mat to store the image that's read
	  Mat img;
	  string fname = "";
	  string pname = "/mnt/export/home/cwmacaul/Documents/vision/data/";
	  char cname[256];
	  printf("\n\nRunmode: Query Image\n\n\n");
	  cout << "Please enter filename of query image in the data folder:\n>";
	  getline( cin, fname);
	  pname = pname + fname;
	  strcpy( cname, pname.c_str());

	  img = imread( cname );
	  
	  if(img.data == NULL) {
	    printf("Unable to read image %s\n", cname );
	    return(0);
	  }
	 
	  printf("\n\n\n\nRead of '%s' successful. \nEnter number of desired top result images:\n", cname);
	  cin >> top_length;
	  // create a window
	  namedWindow(cname, 2);
	  // display query image
	  imshow(cname, img);
	  resizeWindow( cname, 500, 500 );

	  vector<string> results;
	  if (comptype == 1){
	    results = task1_query( img, top_length );
	    displayQandR( results );
	  } else if (comptype == 2){
	    results = task2_query( img, top_length );
	    displayQandR( results );
	  } else if (comptype == 3){
	    results = task3_query( img, top_length );
	    displayQandR( results );
	  } else if (comptype ==4 ){
	    results = task4_query( img, top_length );
	    displayQandR( results );
	  } else if (comptype ==5 ){
	    results = task5_query( img, top_length );
	    displayQandR( results );
	  }
	  waitKey(0);
	  destroyAllWindows();
	}
}









/* displayQandR takes in the top string of result images 
and displays each of them in consistent size*/
void displayQandR( vector<string> results ){
	    
  for ( int i = 0; i < top_length; i++){
    // show the image in a window
    Mat src;
    char fn[500];
    strcpy( fn, results[i].c_str() );
    src = imread( fn );
    if(src.data == NULL) {
      printf("Unable to read result image %s\n", results[i] );
      return;
    }
    namedWindow( fn, 2);
    imshow(fn, src );
    // resizeWindow( fn, src.size().width/8, src.size().height/8 );
    resizeWindow( fn, 500, 500 );
  }
}




	  
/* This function is the query function for the first task. It analyzes the 
   feature vector file and returns an array of complete filenames of images of given length
   that are closest to the query image by sum squared distance. */
vector<string> task1_query( Mat query, int results ){

  // Vector of strings to store the top results:
  vector<string> result_names;
  // Mats to store hsv color version.
  //Mat hsv_img;
  vector<int> qhues;
  // Convert the image to hsv color space
  //cvtColor(query, hsv_img, CV_BGR2HSV );
  
  // For task 1, get the center 25 pixels and keep track of their hues. 
  for(int x = (int)(query.size().width/2)-2; x<=((int)(query.size().width/2)+2); x++ ){
    for(int y = (int)(query.size().height/2)-2; y <=((int)(query.size().height/2)+2); y++ ){  
      Vec3b rgb = query.at<Vec3b>(Point(x,y));
      qhues.push_back((int)rgb.val[0]);
      qhues.push_back((int)rgb.val[1]);
      qhues.push_back((int)rgb.val[2]);
    }
  }

  for( int i = 0; i < 75; i+=3 ) {
	cout << qhues[i];
	cout<<",";
	cout<< qhues[i+1];
	cout<< ",";
	cout << qhues[i+2];
	cout << "\n";
  }
      

  // For task 1, get the center 25 pixels and keep track of their hues. 
  for(int x = (int)(query.size().width/2)-2; x<=((int)(query.size().width/2)+2); x++ ){
    for(int y = (int)(query.size().height/2)-2; y <=((int)(query.size().height/2)+2); y++ ){  
      // get the intensity of BGR at this pixel
      Vec3b intensity = query.at<Vec3b>(Point(x,y));
      // edit the pixel to have maximum blue intensity
      //intf("intensity      %d", intensity);
      intensity.val[0] = 255;
      intensity.val[1] = 255;
      intensity.val[2] = 255;
      query.at<Vec3b>(Point(x,y)) = intensity;
    }
  }

  // this saves the edited (blue-ified) image
  imwrite("../data/edited_query_image.png", query);

  
  
  // The function is hard-coded to find the file 'task1_FV.txt' for the feature vectors.
  ifstream file;
  string line;
  file.open("task1_FV.txt");


  // this is a map to hold the filename and the ssd.
  map<float, string> namessd;
  
  while( getline(file, line))
    {
      stringstream   linestream(line);
      string         name;
      int            rgbs[75];
      char fixname[256];
      vector<int> rhues;

      linestream >> name;
      // getline( linestream, name, "\t" );  // read up-to the first tab (discard tab).

      strcpy(fixname, name.c_str());
      //printf("fixname: %s\n", fixname);

      
      
      // Read the integers using the operator >>
      for (int i = 0; i < 75; i++){
        int val;
        linestream >> rgbs[i];
      }
      float ssd = 0;
      float power = 2.0;
      for (int i = 0; i < 75; i+=3){
        float sd;
	//printf( "qhues - rhues: \t%d\n", abs(qhues[i]-rgbs[i]) );
	sd = ((abs(qhues[i]-rgbs[i]))+(abs(qhues[i+1]-rgbs[i+1]))+(abs(qhues[i+2]-rgbs[i+2])));
	if( i == 0 ){
	cout << fixname;
	cout << "\t";
	cout << rgbs[i];
	cout<<",";
	cout<< rgbs[i+1];
	cout<< ",";
	cout << rgbs[i+2];
	cout << "\n";
	}
	//sd = abs( (float)qhues[i] - (float)rgbs[i] );
	ssd +=  pow(sd,power);
	//printf(" SSD: \t %f\n", ssd);
      }
      namessd[ssd] = fixname;   
    }
  file.close();

  int rc = 0;
  for(map<float, string >::const_iterator it = namessd.begin();
      it != namessd.end(); ++it)
    {
      if (rc < results){
	result_names.push_back( it->second );
        cout << "SSD:\t";
	cout << it->first;
	cout << "\t\tNM:\t";
	cout << it->second.c_str();
	cout << "\n";
	rc++;
      } else {
	break;
      }
    }
  
  return result_names;
  
}














	  
/* This function is the query function for the second task. It analyzes the 
   feature vector file and returns an array of complete filenames of images of given length
   that are closest to the query image by a 2d hue-saturation histogram comparison by intersection. */
vector<string> task2_query( Mat query, int results ){

  Mat hsv_img;
  cvtColor(query, hsv_img, CV_BGR2HSV );
      // Quantize the hue to 30 levels
    // and the saturation to 32 levels
    int hbins = 30, sbins = 30;
    int histSize[] = {hbins, sbins};
    // hue varies from 0 to 179, see cvtColor
    float hranges[] = { 0, 180 };
    // saturation varies from 0 (black-gray-white) to
    // 255 (pure spectrum color)
    float sranges[] = { 0, 256 };
    const float* ranges[] = { hranges, sranges };
    MatND hist;
    // we compute the histogram from the 0-th and 1-st channels
    int channels[] = {0, 1};

    calcHist( &hsv_img, 1, channels, Mat(), // do not use mask
             hist, 2, histSize, ranges,
             true, // the histogram is uniform
             false );


    hist.convertTo( hist, CV_32F );
    normalize( hist, hist, 0, 1, NORM_MINMAX, -1, Mat() );

    // The function is hard-coded to find the file 'task1_FV.txt' for the feature vectors.
    ifstream file;
    string line;
    file.open("task2_FV.txt");

    // map for the comparison and name values
    map<double, string> nameix;

    while( getline(file, line))
      {
	stringstream   linestream(line);
	string         name;
	int            hisvals[75];
	char           fixname[256];
	
	linestream >> name;
	
	strcpy(fixname, name.c_str());

	//	printf( "name: \t\t%s\n", fixname);
	Mat rhist = Mat::zeros(30, 30, CV_32F);//Matrix to store values

	for( int h = 0; h < 30; h++ )
	  for( int s = 0; s < 30; s++ ){
	    float m;
	    linestream >> m;
	    rhist.at<float>(h, s) = m;
	  }
	normalize( rhist, rhist, 0, 1, NORM_MINMAX, -1, Mat() );
	double compare = compareHist( hist, rhist, 2 );
	nameix[compare] = fixname;
      }
    file.close();

    vector<string> result_names;
    int rc = 0;
    for(map<double, string >::reverse_iterator it = nameix.rbegin();
	it != nameix.rend(); ++it)
      {
	if (rc < results){
	  //char cn[256];
	  //strcpy( cn, it->second.c_str() );
	  result_names.push_back( it->second );
	  cout << "BC:\t";
	  cout << it->first;
	  cout << "\t\tNM:\t";
	  cout << it->second.c_str();
	  cout << "\n";
	  //printf( "pushing into result vector: \t\t %s\n", it->second.c_str() );
	  rc++;
	} else {
	  break;
	}
      }
    
    return result_names;
}









/* This function is the query function for the third task. It analyzes the 
   feature vector file and returns an array of complete filenames of images of given length
   that are closest to the query image by a 2d hue-saturation histogram in the center of
   the image and the bottom of the image. Comparison is by intersection. */
vector<string> task3_query( Mat query, int results ){

    // This is the center third in the x dimension and center 1/5 in the y dimension. 
    Mat centerarea = query(Range( query.rows/3, (query.rows/3)*2 ), Range( (query.cols/5)*2, (query.cols/5)*3 ) );
    // Mats to store hsv color version.
    Mat hsv_img;
    // Convert the image to hsv color space
    cvtColor(centerarea, hsv_img, CV_BGR2HSV );
      // Quantize the hue to 30 levels
    // and the saturation to 32 levels
    int hbins = 30, sbins = 30;
    int histSize[] = {hbins, sbins};
    // hue varies from 0 to 179, see cvtColor
    float hranges[] = { 0, 180 };
    // saturation varies from 0 (black-gray-white) to
    // 255 (pure spectrum color)
    float sranges[] = { 0, 256 };
    const float* ranges[] = { hranges, sranges };
    MatND centerhist;
    // we compute the histogram from the 0-th and 1-st channels
    int channels[] = {0, 1};

    calcHist( &hsv_img, 1, channels, Mat(), // do not use mask
             centerhist, 2, histSize, ranges,
             true, // the histogram is uniform
             false );

    centerhist.convertTo( centerhist, CV_32F );
    normalize( centerhist, centerhist, 0, 1, NORM_MINMAX, -1, Mat() );


    // Now this is for the bottom half of the image:
    Mat barea = query(Range( 0, query.rows-1 ), Range( (query.cols/2), query.cols-1 ) );
    Mat hsv_imgb;
    
    // Convert the image to hsv color space
    cvtColor(barea, hsv_imgb, CV_BGR2HSV );
          // Quantize the hue to 30 levels
    // and the saturation to 32 levels
    int hbinsb = 30, sbinsb = 30;
    int histSizeb[] = {hbinsb, sbinsb};
    // hue varies from 0 to 179, see cvtColor
    float hrangesb[] = { 0, 180 };
    // saturation varies from 0 (black-gray-white) to
    // 255 (pure spectrum color)
    float srangesb[] = { 0, 256 };
    const float* rangesb[] = { hranges, sranges };
    MatND bhist;
    // we compute the histogram from the 0-th and 1-st channels
    int channelsb[] = {0, 1};

    calcHist( &hsv_imgb, 1, channelsb, Mat(), // do not use mask
             bhist, 2, histSizeb, rangesb,
             true, // the histogram is uniform
             false );

    bhist.convertTo( bhist, CV_32F );
    normalize( bhist, bhist, 0, 1, NORM_MINMAX, -1, Mat() );

    // The function is hard-coded to find the file 'task1_FV.txt' for the feature vectors.
    ifstream file;
    string line;
    file.open("task3_FV.txt");

    // map for the comparison and name values
    map<double, string> nameix;

    while( getline(file, line))
      {
	stringstream   linestream(line);
	string         name;
	int            hisvals[75];
	char           fixname[256];
	
	linestream >> name;
	
	strcpy(fixname, name.c_str());

	//	printf( "name: \t\t%s\n", fixname);
	Mat chist = Mat::zeros(30, 30, CV_32F);//Matrix to store center values
	Mat bottomhist = Mat::zeros(30, 30, CV_32F);// bottom values
	for( int h = 0; h < 30; h++ )
	  for( int s = 0; s < 30; s++ ){
	    float m;
	    linestream >> m;
	    chist.at<float>(h, s) = m;
	  }
	normalize( chist, chist, 0, 1, NORM_MINMAX, -1, Mat() );

	for( int h = 0; h < 30; h++ )
	  for( int s = 0; s < 30; s++ ){
	    float m;
	    linestream >> m;
	    bottomhist.at<float>(h, s) = m;
	  }
	normalize( bottomhist, bottomhist, 0, 1, NORM_MINMAX, -1, Mat() );
	

	// Compare the center histograms:
	double comparecenter = compareHist( centerhist, chist, 2 );
	// Compare the bottom histograms:
	double comparebottom = compareHist( bhist, bottomhist, 2 );
	nameix[(((comparecenter*.75) + (comparebottom*.25))/2)] = fixname;
      }
    file.close();

    vector<string> result_names;
    int rc = 0;
    for(map<double, string >::reverse_iterator it = nameix.rbegin();
	it != nameix.rend(); ++it)
      {
	if (rc < results){
	  //char cn[256];
	  //strcpy( cn, it->second.c_str() );
	  result_names.push_back( it->second );
	  cout << "Idis:\t";
	  cout << it->first;
	  cout << "\t\tNM:\t";
	  cout << it->second.c_str();
	  cout << "\n";
	  //printf( "pushing into result vector: \t\t %s\n", it->second.c_str() );
	  rc++;
	} else {
	  break;
	}
      }
    
    return result_names;
}












/* This function is the query function for the fourth task. It analyzes the 
   feature vector file and returns an array of complete filenames of images of given length
   that are closest to the query image by a 2d histogram comparison of sobel filters in the 
   x and y direction for kernel sizes 3, 5, and 7 as well as another 2d histogram of hue and 
   saturation. Comparison done by Chi square. */
vector<string> task4_query( Mat query, int results ){
  
    // Mats to store greyscale and sobels in x and y of each kernel size.
    Mat greyscale,sobelx3,sobely3, sobelx5, sobely5, sobelx7, sobely7;
    //convert to greyscale for the sobels
    cvtColor(query, greyscale, CV_BGR2GRAY);
    Sobel(greyscale, sobelx3, CV_32F, 1, 0, 3);
    Sobel(greyscale, sobely3, CV_32F, 0, 1, 3);
    vector<Mat> merge3;
    merge3.push_back(sobelx3);
    merge3.push_back(sobely3);
    Mat merge3d;
    merge(merge3, merge3d);
    Sobel(greyscale, sobelx5, CV_32F, 1, 0, 5);
    Sobel(greyscale, sobely5, CV_32F, 0, 1, 5);
    vector<Mat> merge5;
    merge5.push_back(sobelx5);
    merge5.push_back(sobely5);
    Mat merge5d;
    merge(merge5, merge5d);
    Sobel(greyscale, sobelx7, CV_32F, 1, 0, 7);
    Sobel(greyscale, sobely7, CV_32F, 0, 1, 7);
    vector<Mat> merge7;
    merge7.push_back(sobelx7);
    merge7.push_back(sobely7);
    Mat merge7d;
    merge(merge7, merge7d);


    // 30 bins for x and y sobel filters.
    int hbins = 30, sbins = 30;
    int histSize[] = {hbins, sbins};
    float hranges[] = { 0, 255 };
    float sranges[] = { 0, 255 };
    const float* ranges[] = { hranges, sranges };
    MatND hist3, hist5, hist7, histcolor;
    // we compute the histogram from the 0-th and 1-st channels
    int channels[] = {0, 1};

    // Histogram for sobels with kernel size 3. 
    calcHist( &merge3d, 1, channels, Mat(), // do not use mask
             hist3, 2, histSize, ranges,
	      true, // the histogram is uniform
	      false );
    // Histogram for sobels with kernel size 5. 
    calcHist( &merge5d, 1, channels, Mat(), // do not use mask
	      hist5, 2, histSize, ranges,
	      true, // the histogram is uniform
	      false );

    // Histogram for sobels with kernel size 7. 
    calcHist( &merge7d, 1, channels, Mat(), // do not use mask
	      hist7, 2, histSize, ranges,
	      true, // the histogram is uniform
	      false );


    // Histogram for color information. Can incidentally be run with the same params.
    calcHist( &query, 1, channels, Mat(), // do not use mask
	      histcolor, 2, histSize, ranges,
	      true, // the histogram is uniform
	      false );
    

    // The function is hard-coded to find the file 'task1_FV.txt' for the feature vectors.
    ifstream file;
    string line;
    file.open("task4_FV.txt");
    
    // map for the comparison and name values
    map<double, string> nameix;
    
    while( getline(file, line))
      {
	stringstream   linestream(line);
	string         name;
	char           fixname[256];

	// These will hold the histograms for a given database image.
	Mat rhist3 = Mat::zeros(30, 30, CV_32F);
	Mat rhist5 = Mat::zeros(30, 30, CV_32F);
	Mat rhist7 = Mat::zeros(30, 30, CV_32F);
	Mat rhistcolor = Mat::zeros(30, 30, CV_32F);
	
	linestream >> name;
	
	strcpy(fixname, name.c_str());

	printf("pulling: \t\t%s\n", fixname);

	for( int kernels = 0; kernels <4; kernels++){
	  for( int h = 0; h < hbins; h++ ){
	    for( int s = 0; s < sbins; s++ )
	      {
		float binVal;
		if (kernels == 0){
		  float m;
		  linestream >> m;
		  rhist3.at<float>(h, s) = m;
		} else if (kernels == 1 ){
		  float m;
		  linestream >> m;
		  rhist5.at<float>(h, s) = m;
		} else if (kernels == 2)  {
		  float m;
		  linestream >> m;
		  rhist7.at<float>(h, s) = m;
		} else if (kernels == 3) {
		  float m;
		  linestream >> m;
		  rhistcolor.at<float>(h, s) = m;
		}
	      }
	  }
	  kernels++;
	}

	// compare each of the histograms using intersection.
	double c3 = compareHist( hist3, rhist3, 0 );
	double c5 = compareHist( hist5, rhist5, 0 );
	double c7 = compareHist( hist7, rhist7, 0 );
	double cc = compareHist( histcolor, rhistcolor, 0 );
	double sumcompare = c3+c5+c7+cc;

	cout << sumcompare;
	cout <<"\n";
	nameix[sumcompare] = fixname;
      }
    file.close();

    // Go through the map and get the full names for the top N results.
    vector<string> result_names;
    int rc = 0;
    for(map<double,string>::reverse_iterator it = nameix.rbegin();
	it != nameix.rend(); ++it)
      {
	if (rc < results){
	  result_names.push_back( it->second.c_str() );
	  cout << "Idis:\t";
	  cout << it->first;
	  cout << "\t\tNM:\t";
	  cout << it->second.c_str();
	  cout << "\n";
	  rc++;
	  
	} else {
	  break;
	}
      }
    return result_names;
}

















/* This function is the query function for the fifth task. It analyzes the 
   feature vector file and returns an array of complete filenames of images of given length
   that are closest to the query image by a 2d histogram comparison of sobel filters in the 
   x and y direction for kernel sizes 3, 5, and 7---comparison done by Chi square. Half of the 
   weight of distance consideration is the sum of these histogram comparison distances; the other
   half is the difference in variance between the images.*/
vector<string> task5_query( Mat query, int results ){



  
    // This is the center third in the x and y dimensions.
    Mat centerarea = query(Range( query.rows/3, (query.rows/3)*2 ), Range( (query.cols/3), (query.cols/3)*2 ) );
    // Mats to store greyscale and sobels in x and y of each kernel size.
    Mat greyscale,sobelx3,sobely3, sobelx5, sobely5, sobelx7, sobely7;
    //convert to greyscale for the sobels
    cvtColor(centerarea, greyscale, CV_BGR2GRAY);
    Sobel(greyscale, sobelx3, CV_32F, 1, 0, 3);
    Sobel(greyscale, sobely3, CV_32F, 0, 1, 3);
    vector<Mat> merge3;
    merge3.push_back(sobelx3);
    merge3.push_back(sobely3);
    Mat merge3d;
    merge(merge3, merge3d);
    Sobel(greyscale, sobelx5, CV_32F, 1, 0, 5);
    Sobel(greyscale, sobely5, CV_32F, 0, 1, 5);
    vector<Mat> merge5;
    merge5.push_back(sobelx5);
    merge5.push_back(sobely5);
    Mat merge5d;
    merge(merge5, merge5d);
    Sobel(greyscale, sobelx7, CV_32F, 1, 0, 7);
    Sobel(greyscale, sobely7, CV_32F, 0, 1, 7);
    vector<Mat> merge7;
    merge7.push_back(sobelx7);
    merge7.push_back(sobely7);
    Mat merge7d;
    merge(merge7, merge7d);

    

    // 30 bins for x and y sobel filters.
    int hbins = 30, sbins = 30;
    int histSize[] = {hbins, sbins};
    float hranges[] = { 0, 255 };
    float sranges[] = { 0, 255 };
    const float* ranges[] = { hranges, sranges };
    MatND hist3, hist5, hist7;
    // we compute the histogram from the 0-th and 1-st channels
    int channels[] = {0, 1};


    // Histogram for sobels with kernel size 3. 
    calcHist( &merge3d, 1, channels, Mat(), // do not use mask
             hist3, 2, histSize, ranges,
	      true, // the histogram is uniform
	      false );
    // Histogram for sobels with kernel size 5. 
    calcHist( &merge5d, 1, channels, Mat(), // do not use mask
	      hist5, 2, histSize, ranges,
	      true, // the histogram is uniform
	      false );

    // Histogram for sobels with kernel size 7. 
    calcHist( &merge7d, 1, channels, Mat(), // do not use mask
	      hist7, 2, histSize, ranges,
	      true, // the histogram is uniform
	      false );

        // normalize these puppies so we can do intersection.
    normalize( hist3, hist3, 0, 1, NORM_MINMAX, -1, Mat() );
    normalize( hist5, hist5, 0, 1, NORM_MINMAX, -1, Mat() );
    normalize( hist7, hist7, 0, 1, NORM_MINMAX, -1, Mat() );

        // calculate the color variance in the image.
    // first calculate the mean.
    float total;
    float mean;
    for(int x = 0; x <(int)(query.size().width); x++ ){
      for(int y = 0; y <(int)(query.size().height); y++ ){  
	Vec3b rgb = query.at<Vec3b>(Point(x,y));
	total += (rgb[0]+rgb[1]+rgb[1]);
      }
    }
    mean = total/(float)(query.size().width*query.size().width);
    // then calculate the variance.
    float var;
    for(int x = 0; x <(int)(query.size().width); x++ ){
      for(int y = 0; y <(int)(query.size().height); y++ ){  
	Vec3b rgb = query.at<Vec3b>(Point(x,y));
	float val = (rgb[0]+rgb[1]+rgb[1]);
	var += ((val-mean)*(val-mean));
      }
    }

    

    
    // The function is hard-coded to find the file 'task1_FV.txt' for the feature vectors.
    ifstream file;
    string line;
    file.open("task5_FV.txt");
    
    // map for the comparison and name values
    map<double, string> nameix;
    
    while( getline(file, line))
      {
	stringstream   linestream(line);
	string         name;
	char           fixname[256];

	// These will hold the histograms for a given database image.
	Mat rhist3 = Mat::zeros(30, 30, CV_32F);
	Mat rhist5 = Mat::zeros(30, 30, CV_32F);
	Mat rhist7 = Mat::zeros(30, 30, CV_32F);
	
	linestream >> name;
	
	strcpy(fixname, name.c_str());

	for( int kernels = 0; kernels <4; kernels++){
	  for( int h = 0; h < hbins; h++ ){
	    for( int s = 0; s < sbins; s++ )
	      {
		float binVal;
		if (kernels == 0){
		  float m;
		  linestream >> m;
		  rhist3.at<float>(h, s) = m;
		} else if (kernels == 1 ){
		  float m;
		  linestream >> m;
		  rhist5.at<float>(h, s) = m;
		} else if (kernels == 2)  {
		  float m;
		  linestream >> m;
		  rhist7.at<float>(h, s) = m;
		} 
	      }
	  }
	  kernels++;
	}

	//normalize these puppies so we can do intersection.
	normalize( rhist3, rhist3, 0, 1, NORM_MINMAX, -1, Mat() );
	normalize( rhist5, rhist5, 0, 1, NORM_MINMAX, -1, Mat() );
	normalize( rhist7, rhist7, 0, 1, NORM_MINMAX, -1, Mat() );
	// This value is the variance of the image.
	double variance;
	linestream >> variance;

	// calculate the difference in variance.
	double vdiff;
	vdiff = abs(variance-var);
	

	// compare each of the histograms using intersection.
	double c3 = compareHist( hist3, rhist3, 2 );
	double c5 = compareHist( hist5, rhist5, 2 );
	double c7 = compareHist( hist7, rhist7, 2 );
	double sumcompare = (c3+c5+c7)*0.5 + (1/(vdiff)*0.5);

	nameix[sumcompare] = fixname;
      }
    file.close();

    // Go through the map and get the full names for the top N results.
    vector<string> result_names;
    int rc = 0;
    for(map<double,string>::reverse_iterator it = nameix.rbegin();
	it != nameix.rend(); ++it)
      {
	if (rc < results){
	  result_names.push_back( it->second.c_str() );
	  cout << "Idis:\t";
	  cout << it->first;
	  cout << "\t\tNM:\t";
	  cout << it->second.c_str();
	  cout << "\n";
	  rc++;
	  
	} else {
	  break;
	}
      }
    return result_names;
}











    

/* function that goes through the directory and its subdirectories.
   help with implementation: 
   http://stackoverflow.com/questions/9138866/ ... 
   c-list-all-directories-and-subdirectories-within-in-linux */
void processDirectory( char* directory ){

  
  const string pname = path;
  const string fname = pname + directory;
  const string dr = directory;
  char dopen[256];

  if ( depth == 0 ){
    strcpy( dopen, pname.c_str() );
    path = pname;
  } else {
    strcpy( dopen, fname.c_str() );
    path = fname + "/";
  }
  depth++;

  printf( "procDir: opening %s\n", dopen );

  
  DIR *dir = opendir( dopen );



  if ( NULL == dir ){
    printf( "could not open directory %s \n", dopen );
    return;
  }

  struct dirent *entity = readdir(dir);

  while( entity != NULL ){
    processEntity( entity );
    entity = readdir( dir );
  }

  // gone through this directory now so remove from path:
    if ( depth == 0 ){
      //
    } else if ( (path.length() -1) > dr.length() ) {
      path.resize( path.length() -1 - dr.length() );
  }

  closedir( dir );
}

/* helper function that processes the subdirectories processDirectory
   comes up with */
void processEntity( struct dirent* entity ){
  // so this doesn't take forever in debug:
  if (counter >= 200){
    printf( "counter max\n");
    return;
  }


  // figure out what type of entity (directory or a file)

  // if yes, it's a directory. 
  if ( entity-> d_type == DT_DIR ){
    // need to go past the "." and ".." directories.

    if ( (strcmp(entity->d_name, ".") == 0) || (strcmp(entity->d_name, "..") == 0)){
      return;
    }
    else if ( strstr( entity->d_name, "kmcdonel")){
	//
    }
    else{
    // call to the function that deals with directories again.
    
    processDirectory( entity->d_name );
    return;
    }
  }

  // if we are dealing with a file:
  if ( entity->d_type == DT_REG ){
    if( strstr(entity->d_name, ".jpg") ||
	strstr(entity->d_name, ".JPG") || 
	strstr(entity->d_name, ".png") ||
	strstr(entity->d_name, ".PNG") ||
	strstr(entity->d_name, ".ppm") ||
	strstr(entity->d_name, ".tif") ) {
      printf( "filename: \t%s\n", entity->d_name );
      if ( strcmp( entity->d_name, "IMG_6719.JPG") == 0 || strstr( entity->d_name, " ") ){
	//
      } else {
      processFile(  entity );
      return;
      }
    }
  }

    // printf( "%s is inconsequential.\n", entity->d_name );
}

void processFile( struct dirent* file ) {
  
  if (file == NULL){
    printf( "reached NULL file\n");
    return;
  }

  char complete[256];
  const string pname = path;
  const string compfname = pname + file->d_name;
  strcpy( complete, compfname.c_str());

  

    
  // matrix to store src image.
  Mat src;
  src = imread( complete );

  
  
  if(src.data == NULL) {
    printf("Unable to read image %s\n", complete );
    return;
  }




  // This counter can be uncommented to run shorter FV initializations. 
  //  counter++;

  

  // write out the full path to this image so it can be easily read by the analyzer.
  myfile1 << (string)complete;
  myfile1 << "\t";


  
  // For task 1, the feature vectors file is the hue at each of the pixels.
  if ( comptype == 1 ){
    

    // For task 1, get the center 25 pixels and keep track of their hues. 
    for(int x = (int)(src.size().width/2)-2; x <=((int)(src.size().width/2)+2); x++ ){
      for(int y = (int)(src.size().height/2)-2; y <=((int)(src.size().height/2)+2); y++ ){  
	Vec3b rgb = src.at<Vec3b>(Point(x,y));
	// int hue = hsv[0];
	//ostringstream temp;
	//temp<<hue;
	//myfile1 << temp.str();
	myfile1 << (int)rgb.val[0];
	myfile1 << "\t";
	myfile1 << (int)rgb.val[1];
	myfile1 << "\t";
	myfile1 << (int)rgb.val[2];
	myfile1 << "\t";
      }
    }    
      myfile1 << "\n";
  }
  else if ( comptype == 2 ){

    
    // Mats to store hsv color version.
    Mat hsv_img;
    // Convert the image to hsv color space
    cvtColor(src, hsv_img, CV_BGR2HSV );

    // Quantize the hue to 30 levels
    // and the saturation to 32 levels
    int hbins = 30, sbins = 30;
    int histSize[] = {hbins, sbins};
    // hue varies from 0 to 179, see cvtColor
    float hranges[] = { 0, 180 };
    // saturation varies from 0 (black-gray-white) to
    // 255 (pure spectrum color)
    float sranges[] = { 0, 256 };
    const float* ranges[] = { hranges, sranges };
    MatND hist;
    // we compute the histogram from the 0-th and 1-st channels
    int channels[] = {0, 1};

    calcHist( &hsv_img, 1, channels, Mat(), // do not use mask
             hist, 2, histSize, ranges,
             true, // the histogram is uniform
             false );

    for( int h = 0; h < hbins; h++ )
      for( int s = 0; s < sbins; s++ )
        {
    	  float binVal = hist.at<float>(h, s);
    	  myfile1 <<binVal;
    	  myfile1 <<"\t";
        }
    myfile1 << "\n";
  }

  else if ( comptype == 3 ) {

    // This is the center third in the x dimension and center 1/5 in the y dimension. 
    Mat centerarea = src(Range( src.rows/3, (src.rows/3)*2 ), Range( (src.cols/5)*2, (src.cols/5)*3 ) );
        // Mats to store hsv color version.
    Mat hsv_img;
    // Convert the image to hsv color space
    cvtColor(centerarea, hsv_img, CV_BGR2HSV );

    // Quantize the hue to 30 levels
    // and the saturation to 30 levels
    int hbins = 30, sbins = 30;
    int histSize[] = {hbins, sbins};
    // hue varies from 0 to 179, see cvtColor
    float hranges[] = { 0, 180 };
    // saturation varies from 0 (black-gray-white) to
    // 255 (pure spectrum color)
    float sranges[] = { 0, 256 };
    const float* ranges[] = { hranges, sranges };
    MatND hist;
    // we compute the histogram from the 0-th and 1-st channels
    int channels[] = {0, 1};

    calcHist( &hsv_img, 1, channels, Mat(), // do not use mask
             hist, 2, histSize, ranges,
             true, // the histogram is uniform
             false );

    for( int h = 0; h < hbins; h++ )
      for( int s = 0; s < sbins; s++ )
        {
    	  float binVal = hist.at<float>(h, s);
    	  myfile1 <<binVal;
    	  myfile1 <<"\t";
        }

    // Now this is for the bottom half of the image:
    Mat barea = src(Range( 0, src.rows-1 ), Range( (src.cols/2), src.cols-1 ) );
    Mat hsv_imgb;
    
    // Convert the image to hsv color space
    cvtColor(barea, hsv_imgb, CV_BGR2HSV );

          // Quantize the hue to 30 levels
    // and the saturation to 32 levels
    int hbinsb = 30, sbinsb = 30;
    int histSizeb[] = {hbinsb, sbinsb};
    // hue varies from 0 to 179, see cvtColor
    float hrangesb[] = { 0, 180 };
    // saturation varies from 0 (black-gray-white) to
    // 255 (pure spectrum color)
    float srangesb[] = { 0, 256 };
    const float* rangesb[] = { hranges, sranges };
    MatND bhist;
    // we compute the histogram from the 0-th and 1-st channels
    int channelsb[] = {0, 1};

    calcHist( &hsv_imgb, 1, channelsb, Mat(), // do not use mask
             bhist, 2, histSizeb, rangesb,
             true, // the histogram is uniform
             false );

    for( int h = 0; h < hbins; h++ )
      for( int s = 0; s < sbins; s++ )
        {
    	  float binVal = bhist.at<float>(h, s);
    	  myfile1 <<binVal;
    	  myfile1 <<"\t";
        }

    
    myfile1 << "\n";
    
  
  }
  else if ( comptype == 4 ) {
    
    // Mats to store greyscale and sobels in x and y of each kernel size.
    Mat greyscale,sobelx3,sobely3, sobelx5, sobely5, sobelx7, sobely7;
    //convert to greyscale for the sobels
    cvtColor(src, greyscale, CV_BGR2GRAY);
    Sobel(greyscale, sobelx3, CV_32F, 1, 0, 3);
    Sobel(greyscale, sobely3, CV_32F, 0, 1, 3);
    vector<Mat> merge3;
    merge3.push_back(sobelx3);
    merge3.push_back(sobely3);
    Mat merge3d;
    merge(merge3, merge3d);
    Sobel(greyscale, sobelx5, CV_32F, 1, 0, 5);
    Sobel(greyscale, sobely5, CV_32F, 0, 1, 5);
    vector<Mat> merge5;
    merge5.push_back(sobelx5);
    merge5.push_back(sobely5);
    Mat merge5d;
    merge(merge5, merge5d);
    Sobel(greyscale, sobelx7, CV_32F, 1, 0, 7);
    Sobel(greyscale, sobely7, CV_32F, 0, 1, 7);
    vector<Mat> merge7;
    merge7.push_back(sobelx7);
    merge7.push_back(sobely7);
    Mat merge7d;
    merge(merge7, merge7d);


    // 30 bins for x and y sobel filters.
    int hbins = 30, sbins = 30;
    int histSize[] = {hbins, sbins};
    float hranges[] = { 0, 255 };
    float sranges[] = { 0, 255 };
    const float* ranges[] = { hranges, sranges };
    MatND hist3, hist5, hist7, histcolor;
    // we compute the histogram from the 0-th and 1-st channels
    int channels[] = {0, 1};

    // Histogram for sobels with kernel size 3. 
    calcHist( &merge3d, 1, channels, Mat(), // do not use mask
             hist3, 2, histSize, ranges,
	      true, // the histogram is uniform
	      false );
    // Histogram for sobels with kernel size 5. 
    calcHist( &merge5d, 1, channels, Mat(), // do not use mask
	      hist5, 2, histSize, ranges,
	      true, // the histogram is uniform
	      false );

    // Histogram for sobels with kernel size 7. 
    calcHist( &merge7d, 1, channels, Mat(), // do not use mask
	      hist7, 2, histSize, ranges,
	      true, // the histogram is uniform
	      false );


    // Histogram for color information. Can incidentally be run with the same params.
    calcHist( &src, 1, channels, Mat(), // do not use mask
	      histcolor, 2, histSize, ranges,
	      true, // the histogram is uniform
	      false );

    
    
    for( int kernels = 0; kernels <4; kernels++){
      for( int h = 0; h < hbins; h++ ){
	for( int s = 0; s < sbins; s++ )
	  {
	    float binVal;
	    if (kernels == 0){
	      binVal = hist3.at<float>(h, s);
	    } else if (kernels == 1 ){
	      binVal = hist5.at<float>(h, s);
	    } else if (kernels == 2)  {
	      binVal = hist7.at<float>(h, s);
	    } else if (kernels == 3){
	      binVal = histcolor.at<float>(h, s);
	    }
	    myfile1 <<binVal;
	    myfile1 <<"\t";
	  }
      }
      kernels++;
    }
  } else if ( comptype == 5 ) {
    
    // This is the center third in the x and y dimensions.
    Mat centerarea = src(Range( src.rows/3, (src.rows/3)*2 ), Range( (src.cols/3), (src.cols/3)*2 ) );
    // Mats to store greyscale and sobels in x and y of each kernel size.
    Mat greyscale,sobelx3,sobely3, sobelx5, sobely5, sobelx7, sobely7;
    //convert to greyscale for the sobels
    cvtColor(centerarea, greyscale, CV_BGR2GRAY);
    Sobel(greyscale, sobelx3, CV_32F, 1, 0, 3);
    Sobel(greyscale, sobely3, CV_32F, 0, 1, 3);
    vector<Mat> merge3;
    merge3.push_back(sobelx3);
    merge3.push_back(sobely3);
    Mat merge3d;
    merge(merge3, merge3d);
    Sobel(greyscale, sobelx5, CV_32F, 1, 0, 5);
    Sobel(greyscale, sobely5, CV_32F, 0, 1, 5);
    vector<Mat> merge5;
    merge5.push_back(sobelx5);
    merge5.push_back(sobely5);
    Mat merge5d;
    merge(merge5, merge5d);
    Sobel(greyscale, sobelx7, CV_32F, 1, 0, 7);
    Sobel(greyscale, sobely7, CV_32F, 0, 1, 7);
    vector<Mat> merge7;
    merge7.push_back(sobelx7);
    merge7.push_back(sobely7);
    Mat merge7d;
    merge(merge7, merge7d);


    // 30 bins for x and y sobel filters.
    int hbins = 30, sbins = 30;
    int histSize[] = {hbins, sbins};
    float hranges[] = { 0, 255 };
    float sranges[] = { 0, 255 };
    const float* ranges[] = { hranges, sranges };
    MatND hist3, hist5, hist7;
    // we compute the histogram from the 0-th and 1-st channels
    int channels[] = {0, 1};

    // Histogram for sobels with kernel size 3. 
    calcHist( &merge3d, 1, channels, Mat(), // do not use mask
             hist3, 2, histSize, ranges,
	      true, // the histogram is uniform
	      false );
    // Histogram for sobels with kernel size 5. 
    calcHist( &merge5d, 1, channels, Mat(), // do not use mask
	      hist5, 2, histSize, ranges,
	      true, // the histogram is uniform
	      false );

    // Histogram for sobels with kernel size 7. 
    calcHist( &merge7d, 1, channels, Mat(), // do not use mask
	      hist7, 2, histSize, ranges,
	      true, // the histogram is uniform
	      false );

    
    
    for( int kernels = 0; kernels <4; kernels++){
      for( int h = 0; h < hbins; h++ ){
	for( int s = 0; s < sbins; s++ )
	  {
	    float binVal;
	    if (kernels == 0){
	      binVal = hist3.at<float>(h, s);
	    } else if (kernels == 1 ){
	      binVal = hist5.at<float>(h, s);
	    } else if (kernels == 2)  {
	      binVal = hist7.at<float>(h, s);
	    }
	    myfile1 <<binVal;
	    myfile1 <<"\t";
	  }
      }
      kernels++;
    }

    // calculate the color variance in the image.
    // first calculate the mean.
    float total;
    float mean;
    for(int x = 0; x <(int)(src.size().width); x++ ){
      for(int y = 0; y <(int)(src.size().height); y++ ){  
	Vec3b rgb = src.at<Vec3b>(Point(x,y));
	total += (rgb[0]+rgb[1]+rgb[1]);
      }
    }
    mean = total/(float)(src.size().width*src.size().width);
    // then calculate the variance.
    float var;
    for(int x = 0; x <(int)(src.size().width); x++ ){
      for(int y = 0; y <(int)(src.size().height); y++ ){  
	Vec3b rgb = src.at<Vec3b>(Point(x,y));
	float val = (rgb[0]+rgb[1]+rgb[1]);
	var += ((val-mean)*(val-mean));
      }
    }
    myfile1 << var;
    myfile1 << "\t";
  }
  

  myfile1 << "\n";    
}

  

    
	
