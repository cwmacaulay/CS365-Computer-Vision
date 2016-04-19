/*
	Bruce A. Maxwell
	J16 
	Simple example of reading, manipulating, displaying, and writing an image

	Compile command

	clang++ -o imod -I /opt/local/include imgMod.cpp -L /opt/local/lib -lopencv_core -lopencv_highgui 

*/

#include <cstdio>
#include <cstring>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
using namespace cv;


/* This function sets the blue value at each pixel of an
   input image to the maximum value of 255 and saves the 
   altered image as 'edited_image' in the data directory.*/
int task1( char *filename){
  Mat src;
  // read the image
  src = cv::imread(filename);
  // test if the read was successful
  if(src.data == NULL) {
    printf("Unable to read image %s\n", filename);
    exit(-1);
  }

  // print out information about the image
  printf("filename:         %s\n", filename);
  printf("Image size:       %d rows x %d columns\n", (int)src.size().height, (int)src.size().width);
  printf("Image dimensions: %d\n", (int)src.channels());
  printf("Image depth:      %d bytes/channel\n", (int)src.elemSize()/src.channels());

  // create a window
  cv::namedWindow(filename, 1);

  // edit the source image here
  // loop through the pixels
  for( int  height=0; height<=(int)src.size().height;height++){
    for(int width =0; width <=(int)src.size().width ;width++ ){
      // get the intensity of BGR at this pixel
      Vec3b intensity = src.at<Vec3b>(Point(height,width));
      // edit the pixel to have maximum blue intensity
      //intf("intensity      %d", intensity);
      intensity.val[0] = 255;
      src.at<Vec3b>(Point(height,width)) = intensity;
    }
  }

  // this saves the edited (blue-ified) image
  imwrite("../data/edited_image.png", src);
	    

  // show the image in a window
  cv::imshow(filename, src);

  // wait for a key press (indefinitely)
  cv::waitKey(0);

  // get rid of the window
  cv::destroyWindow(filename);

  // terminate the program
  printf("Terminating\n");

  return(0);
}

/* This function splits the BGR channels of an input image, removes the 
blue and green channels, converts to grayscale, then does a threshold 
operation which will convert pixels that have a threshold red value of
10 to white. */
int task2( char *filename){
  Mat src, src_grey, dst;
  // read the image
  src = cv::imread(filename);


  // printf( "THE FILENAME FOR IMREAD IS: \t %s", filename);

  
  // test if the read was successful
  if(src.data == NULL) {
    printf("Unable to read image %s\n", filename);
    exit(-1);
  }

  // print out information about the image
  printf("filename:         %s\n", filename);
  printf("Image size:       %d rows x %d columns\n", (int)src.size().height, (int)src.size().width);
  printf("Image dimensions: %d\n", (int)src.channels());
  printf("Image depth:      %d bytes/channel\n", (int)src.elemSize()/src.channels());

  // create a window
  cv::namedWindow(filename, 1);

  // edit the source image here: threshold
  // split the image into color channels
  Mat channel[3];
  split(src  , channel);
  // set green channel to zero.
  channel[1] = Mat::zeros(src.rows, src.cols, CV_8UC1);
  // set blue  channel to zero.
  channel[0] = Mat::zeros(src.rows, src.cols, CV_8UC1);
  // merge all three channels together.
  merge(channel,3,src);
  
  cvtColor( src, src_grey, CV_BGR2GRAY);
  threshold( src_grey, dst,10, 255,THRESH_BINARY);

  // this saves the edited (blue-ified) image
  imwrite("../data/edited_image_task2.png", dst);
	    

  // show the image in a window
  cv::imshow(filename, dst);

  // wait for a key press (indefinitely)
  cv::waitKey(0);

  // get rid of the window
  cv::destroyWindow(filename);

  // terminate the program
  printf("Terminating\n");

  return(0);
}




/* this function mirrors an input image horizontally if 
   the user presses the key 'h' */
int task3( char *filename){
  Mat src, dst;
  // read the image
  src = cv::imread(filename);
  // test if the read was successful
  if(src.data == NULL) {
    printf("Unable to read image %s\n", filename);
    exit(-1);
  }

  // print out information about the image
  printf("filename:         %s\n", filename);
  printf("Image size:       %d rows x %d columns\n", (int)src.size().height, (int)src.size().width);
  printf("Image dimensions: %d\n", (int)src.channels());
  printf("Image depth:      %d bytes/channel\n", (int)src.elemSize()/src.channels());

  // create a window
  cv::namedWindow(filename, 1);

  // show the image in a window
  cv::imshow(filename, src);

  int b;

  // wait for a key press (indefinitely)
  while(true){
    
    b = waitKey(0);
    printf("key %d", b);

    // as found on the linux computer by testing:
    // uparrow is 65362
    // downarrow  is 65364
    // 65363 right
    // left 65361
    // esc is 27
    if ((b == 65362) || (b== 65364)){
      Mat dst;
      flip(src, dst, 0);
      cv::imshow(filename, dst);
      src = dst;
    }
    else if ((b == 65363) || (b == 65361)){
      Mat dst;
      flip(src, dst, 1);
      cv::imshow(filename, dst);
      src = dst;
    }
    else if (b == 27 ){
      // get rid of the window
      cv::destroyWindow(filename);

      // terminate the program
      printf("Terminating\n");

      return(0);
    }
  }
}



/* this function takes in an image and provides the user 
an interface with a trackbar to apply a gaussian filter 
with varying kernel size*/
int task4( char *filename){
  Mat src, dst;
  // read the image
  src = cv::imread(filename);
  // test if the read was successful
  if(src.data == NULL) {
    printf("Unable to read image %s\n", filename);
    exit(-1);
  }

  // print out information about the image
  printf("filename:         %s\n", filename);
  printf("Image size:       %d rows x %d columns\n", (int)src.size().height, (int)src.size().width);
  printf("Image dimensions: %d\n", (int)src.channels());
  printf("Image depth:      %d bytes/channel\n", (int)src.elemSize()/src.channels());

  // create a window
  cv::namedWindow(filename, 1);

  // this creates the slidebar to change the gaussian kernel size
  int slidGausKS = 0;
  createTrackbar( "Guassian Kernel Size", filename, &slidGausKS, 100);

  while (true){
    // actually changes the guassian kernel size in the image
    Mat dst;
    int gauKS =( slidGausKS +1 );
    // this doesn't work if the kernel is even by even dimensions,
    // so let's avoid that.
    if (gauKS > 0)
      gauKS +=( gauKS-1 );
    
    printf("blur kernel size:\t %d \n", gauKS);
    GaussianBlur( src, dst, Size(gauKS, gauKS), 0, 0);
    
    // show the image with the altered kernel size
    imshow( filename, dst );

    // wait for the user to press a key
    int keypress = waitKey(50);

    // if it's the escape key then exit the program
    if (keypress == 27 ){
      // get rid of the window
      cv::destroyWindow(filename);
      // terminate the program
      printf("Terminating\n");
      break;

    }
  }
  return(0);
}



// GLOBAL VARIABLES FOR FACE DETECTION
String face_casc_n = "haarcascade_frontalface_alt.xml";
String eyes_casc_n = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
RNG rng(12345);

/* this function is an organized version of the above
tasks such that there is a neat user interface with
slide bars and mirror capability */
int facedetect_extension( char *filename){
  Mat src;
  // read the image
  src = cv::imread(filename);
  
  // test if the read was successful
  if(src.data == NULL) {
    printf("Unable to read image %s\n", filename);
    exit(-1);
  }

  // print out information about the image
  printf("filename:         %s\n", filename);
  printf("Image size:       %d rows x %d columns\n", (int)src.size().height, (int)src.size().width);
  printf("Image dimensions: %d\n", (int)src.channels());
  printf("Image depth:      %d bytes/channel\n", (int)src.elemSize()/src.channels());

  // create a window
  cv::namedWindow(filename, 1);

  // load the cascades:
  if (!face_cascade.load( face_casc_n ) ){ printf("Error loading\n"); return -1;}
  if (!eyes_cascade.load( eyes_casc_n ) ){ printf("Error loading\n"); return -1;}

  std::vector<Rect> faces;
  // convert to greyscale first. 
  Mat grey;
  cvtColor( src, grey, CV_BGR2GRAY );
  equalizeHist( grey, grey );

  // detect faces
  face_cascade.detectMultiScale( grey, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30,30) );

  for (size_t i = 0; i < faces.size(); i++){

    Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
    ellipse(src, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0 );

    Mat faceROI = grey( faces[i] );
    std::vector<Rect> eyes;

    // this detects the eyes in each face.

    eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30,30));
    
    for( size_t j = 0; j < eyes.size(); j++ ){
      Point center( faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y +  eyes[j].height*0.5);
      int rad = cvRound( (eyes[j].width + eyes[j].height)*0.25);
      circle( src, center, rad, Scalar( 255, 0, 0 ), 4, 8, 0);
    }
  }
  // display the image with the annotations of the face and eye detection.
  imshow( filename, src );

  int keypress = waitKey(50000);
    // get rid of the window
    cv::destroyWindow(filename);
    // terminate the program
    printf("Terminating\n");
  

  return(0);
}


int main(int argc, char *argv[]) {
  cv::Mat src, src_grey, dst;
	char filename[256];

	// usage
	if(argc < 2) {
		printf("Usage %s <image filename>\n", argv[0]);
		exit(-1);
	}
	strcpy(filename, argv[1]);

	// The task to be carried out can be un-commented.
	
	//	task1( filename );
		task2( filename );
	//	task3( filename );
	//	task4( filename );
	//	facedetect_extension( filename );
	
}
