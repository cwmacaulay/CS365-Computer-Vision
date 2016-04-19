/* Charles Macaulay
 04-02-2016
 CS 365
 
 OR.cpp
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
#include <opencv2/opencv.hpp>
#include <map>
#include "segment.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>



using namespace cv;
using namespace std;
using namespace cv::ml;



// function declarations
void insertFV( string name, vector<double> features );
string queryFV( vector<double> features );
int checkFVDB();

// Global stores the distance evaluation chosen.
string dmetric;
// Global stores the number of objects in the FV DB.
int entries;
// Global stores the most recent values being queried.
// vector< vector<double> >            valmatrix;



/*
 Bruce A. Maxwell
 
 This program is an implementation of a 2-pass segmentation algorithm
 on a binary OpenCV 8U1 image. 0 is background, not zero is foreground.
 */

#include <stdlib.h>
#include <stdio.h>
#include "opencv2/opencv.hpp"
#include "segment.h"

// uncomment to generate some debug statements
// #define DEBUG
// #define DEBUGeuc
// #define DEBUGavpts

/*
 */
int locateRegions(cv::Mat &src,
                  cv::Mat &regmap,
                  const long sizeThresh,
                  cv::Mat &centroid,
                  cv::Mat &bbox,
                  cv::Mat &size,
                  const int MaxLocations
                  )
{
    // statics to reduce setup time when calling the function a lot
    static long *boundto = NULL;
    static long *regcard = NULL;
    
    // local variables
    const long imageSize = src.rows * src.cols;
    long i, j, k;
    unsigned char backpix, uppix, curpix;
    long nRegions=0;
    long *bt, *reg;
    unsigned long upID, backID;
    long *sizeid;
    long numLocations = MaxLocations;
    
    // static variable allocations
    if(boundto == NULL)
        boundto = (long *)malloc(sizeof(long) * imageSize/4);
    if(regcard == NULL)
        regcard = (long *)malloc(sizeof(long) * imageSize/4);
    
    // dynamic variable allocations
    sizeid = (long *)malloc(sizeof(long) * MaxLocations );
    
    // allocate the region map (doesn't allocate more than once if already the right size)
    // initialize the region map to -1
    regmap.create( src.rows, src.cols, CV_32SC1);
    regmap = -1;
    
    // initialize the region map and boundto variables
#ifdef DEBUG
    printf("Initializing arrays\n");
#endif
    reg = regcard;
    bt = boundto;
    for(i=0;i<imageSize/4;i++, reg++, bt++) {
        *reg = 0;
        *bt = i;
    }
    
    
#ifdef DEBUG
    printf("Starting 2-pass algorithm\n");
#endif
    
    // segment the image using a 2-pass algorithm (4-connected)
    for(i=0;i<src.rows;i++) {
        backpix = 0; // outside the image is 0
        
        for(j=0;j<src.cols;j++) {
            
            // get the value of the current pixel
            curpix = src.at<unsigned char>(i, j);
            
            // if the current pixel is foreground
            if(curpix) {
                
                // get the up pixel
                uppix = i > 0 ? src.at<unsigned char>(i-1, j) : 0;
                
                // test if up-pixel is foreground
                if(uppix) {
                    
                    // test if back-pixel is foreground
                    if(backpix) {
                        
                        // similar to back pixel as well so get the region IDs
                        upID = regmap.at<int>(i-1, j);
                        backID = regmap.at<int>(i, j-1);
                        
                        if(backID == upID) { // equal region ids
                            regmap.at<int>(i, j) = upID;
                        }
                        else { // not equal region ids
                            
                            if(boundto[upID] < boundto[backID]) {
                                regmap.at<int>(i, j) = boundto[upID];
                                boundto[backID] = boundto[upID];
                            }
                            else {
                                regmap.at<int>(i, j) = boundto[backID];
                                boundto[upID] = boundto[backID];
                            }
                        }
                    }
                    else {
                        // similar only to the top pixel
                        regmap.at<int>(i, j) = regmap.at<int>(i-1, j);
                    }
                }
                else if(backpix) {
                    // similar only to back pixel
                    regmap.at<int>(i, j) = regmap.at<int>(i, j-1);
                }
                else {
                    // not similar to either pixel
                    regmap.at<int>(i, j) = nRegions++;
                }
            }
            
            backpix = curpix;
        }
    }
    
    
    // get out if there's nothing else to do
    if(nRegions == 0) {
#ifdef DEBUG
        printf("No regions\n");
#endif
        return(0);
    }
    
#ifdef DEBUG
    printf("Fixing IDs\n");
#endif
    // second pass, fix the IDs and calculate the region sizes
    for(i=0;i<regmap.rows;i++) {
        for(j=0;j<regmap.cols;j++) {
            if(regmap.at<int>(i, j) >= 0) {
                regmap.at<int>(i, j) = boundto[regmap.at<int>(i, j)];
                
                // need to follow the tree in some special cases
                while(boundto[regmap.at<int>(i, j)] != regmap.at<int>(i, j))
                    regmap.at<int>(i, j) = boundto[regmap.at<int>(i, j)];
                
                regcard[regmap.at<int>(i, j)]++;
            }
        }
    }
    
#ifdef DEBUG
    printf("Calculating the N largest regions\n");
#endif
    
    size.create(numLocations, 1, CV_32SC1);
    
    // grab the N largest ones
    for(i=0;i<MaxLocations;i++) {
        size.at<int>(i, 0) = 0;
        sizeid[i] = -1;
        for(j=0;j<nRegions;j++) {
            
            // don't consider regions already in the list
            for(k=0;k<i;k++) {
                if(j == sizeid[k])
                    break;
            }
            if(k < i)
                continue;
            
            if((regcard[j] > sizeThresh) && (regcard[j] > size.at<int>(i, 0))) {
                size.at<int>(i, 0) = regcard[j];
                sizeid[i] = j;
            }
        }
        if(size.at<int>(i, 0) == 0) {
            break;
        }
    }
    
    
    numLocations = i;
    
#ifdef DEBUG
    printf("Calculating centroids for %ld regions\n", numLocations);
    for(i=0;i<numLocations;i++) {
        printf("id %ld size %d\n", sizeid[i], size.at<int>(i, 0));
    }
#endif
    
    // now calculate centroids and bounding boxes
    bbox.create(numLocations, 4, CV_32SC1);
    
    centroid.create(numLocations, 2, CV_32SC1);
    centroid = Scalar::all(0);
    
    for(i=0;i<numLocations;i++) {
        bbox.at<int>(i, 0) = bbox.at<int>(i, 1) = 10000;
        bbox.at<int>(i, 2) = bbox.at<int>(i, 3) = 0;
    }
    
    for(j=0;j<src.rows;j++) {
        for(k=0;k<src.cols;k++) {
            for(i=0;i<numLocations;i++) {
                if(regmap.at<int>(j, k) == sizeid[i]) {
                    centroid.at<int>(i, 0) += j; // rows
                    centroid.at<int>(i, 1) += k; // columns
                    
                    bbox.at<int>(i, 0) = j < bbox.at<int>(i, 0) ? j : bbox.at<int>(i, 0);
                    bbox.at<int>(i, 1) = k < bbox.at<int>(i, 1) ? k : bbox.at<int>(i, 1);
                    bbox.at<int>(i, 2) = j > bbox.at<int>(i, 2) ? j : bbox.at<int>(i, 2);
                    bbox.at<int>(i, 3) = k > bbox.at<int>(i, 3) ? k : bbox.at<int>(i, 3);
                    
                    regmap.at<int>(j, k) = i;
                    break;
                }
            }
            if(i == numLocations) {
                regmap.at<int>(j, k) = -1;
            }
        }
    }
    
    for(i=0;i<numLocations;i++) {
        centroid.at<int>(i, 0) /= size.at<int>(i, 0);
        centroid.at<int>(i, 1) /= size.at<int>(i, 0);
    }
    
#ifdef DEBUG
    printf("Terminating normally\n");
#endif
    
    free(sizeid);
    
    return(numLocations);
}

int main(int argc, char *argv[]) {
    
    // initialize distance metric.
    dmetric = "Euclidean";
    // initialize the number of entries by checking FV DB.
    entries = checkFVDB();
    
    printf("\nAt any time, enter a new distance metric:\n[1]\t\tEuclidean\n[2]\t\tKNN\n");
    
    VideoCapture *capdev;
    
    // open the video device
    capdev = new VideoCapture(0);
    if( !capdev->isOpened() ){
        printf("Unable to open video device\n");
        return(-1);
    }
    
    namedWindow("normal", 1);
    namedWindow("thresh", 1);
    namedWindow("hsv", 1);
    
    for(;;) {
        Mat frame, gaus, hsv, grey, thresh, er, dil;
        
#ifdef DEBUG
        printf("Starting loop.\n");
#endif
        *capdev >> frame; // get a new frame from the camera, treat as a stream
        
        GaussianBlur( frame, gaus, Size(3, 3), 0, 0);
        cvtColor( gaus, hsv, CV_BGR2HSV );
        cvtColor( gaus, grey, CV_BGR2GRAY);
        threshold( grey, thresh,120, 255,THRESH_BINARY_INV);
        Mat element = getStructuringElement( MORPH_RECT,
                                            Size( 2*5 + 1, 2*5+1 ),
                                            Point( 5, 5 ) );
        Mat elementsmaller = getStructuringElement( MORPH_RECT,
                                                   Size( 2*2 + 1, 2*2+1 ),
                                                   Point( 2, 2 ) );
    
        for( int i = 0; i < 7; i++){
            erode( thresh, er, elementsmaller );
        }
        
        for( int i = 0; i < 20; i++){
            erode( er, dil, elementsmaller );
            dilate(er, dil, elementsmaller );
            dilate(er, dil, element );
            dilate(er, dil, element );
        }
        for( int i = 0; i < 20; i++){
            erode(er, dil, elementsmaller );
        }
        for( int i = 0; i < 60; i++){
            dilate(er, dil, element );
        }
#ifdef DEBUG
        printf("Morphological processes completed.\n");
#endif
        
        // This is the output matrix with the labelled components.
        Mat outcomp(dil.size(), CV_32S);
        // This holds the centroid.
        Mat centroid;
        // Holds bounding box matrix.
        Mat bbox;
        // Holds size matrix.
        Mat size;
        // sizeThresh
        const long sizeThresh = 10;
        // max locs
        const long maxlocs = 10;
        
        
        int concomp = locateRegions( dil, outcomp, sizeThresh, centroid, bbox, size, maxlocs );
#ifdef DEBUG
        printf("Regions located.\n");
#endif
        
        // int concomp = connectedComponents( dil, outcomp, 8);
        // Find the component that is closest to the center of the image.
        map<int, int> lablocs;
        pair<double,double> avgpt = make_pair(0,0);
        int curlab;
        int counter = 0;
        double minDistFromCtr = 1000.0;
        int minDistLab = 0;
        double avgr;
        double avgc;
        for(int r = 0; r < outcomp.rows; ++r){
            for(int c = 0; c < outcomp.cols; ++c){
                int label = outcomp.at<int>(r, c);
                Vec3b huesi = hsv.at<Vec3b>(Point(c,r));
                huesi.val[0] = label * 80%255;
                huesi.val[1] = label * 80%255;
                huesi.val[2] = label * 80%255;
                hsv.at<Vec3b>(Point(c,r)) = huesi;
                if( (r == 0) && ( c==0) ){
                    curlab = label;
                }
                
                // don't consider the background.
                if( label > -1 ){
                    if ( curlab == label ){
                        avgpt.first += r;
                        avgpt.second += c;
                        counter++;
                    } else {
                        avgr = avgpt.first/counter;
                        avgc = avgpt.second/counter;
                        double distfromctr = pow(( pow((double(outcomp.rows/2)-avgr),2.0) + pow((double(outcomp.cols/2)-avgc),2.0)), 0.5);
                        if  ((distfromctr <= minDistFromCtr)) {
                            minDistFromCtr = distfromctr;
                            minDistLab = curlab;
                        }
                        avgpt.first = 0;
                        avgpt.second = 0;
                        counter = 0;
                        curlab = label;
                    }
                }
            }
        }
        
#ifdef DEBUGavpts
        // if the avg point has been calculated for the component, show a dot around it on the image.
        // edit: this only actually does it for the last one looked at, not necessarily the central one.
        if( avgpt.first > 0 ){
            for ( int d = -1; d <1;d++ ){
                for ( int k = -1; k<1; k++ ){
                    Vec3b huesi = hsv.at<Vec3b>(Point(int(avgc)+d,int(avgr)+k));
                    huesi.val[0] = 6;
                    huesi.val[1] = 170;;
                    huesi.val[2] = 255;
                    frame.at<Vec3b>(Point(int(avgc)+d,int(avgr)+k)) = huesi;
                }
            }
        }
        printf("Most central component with ID %d found.\n", minDistLab);
#endif
        // Build a vector of points for the center component.
        // Also get the average hue
        vector<Point> vpts;
        double avghue = 0;
        double avgsat = 0;
        int pctr = 0;
        for(int r = 0; r < outcomp.rows; ++r){
            for(int c = 0; c < outcomp.cols; ++c){
                int label = outcomp.at<int>(r, c);
                if (label == minDistLab){
                    vpts.push_back( Point(c,r) );
                    Vec3b huesi = hsv.at<Vec3b>(Point(c,r));
                    avghue += huesi.val[0];
                    avgsat += huesi.val[1];
                    pctr++;
                    huesi.val[0] = 90;
                    huesi.val[1] = 0;
                    huesi.val[2] = 255;
                    thresh.at<Vec3b>(Point(c,r)) = huesi;
                }
            }
        }
        
#ifdef DEBUG
        printf("%d Points gathered that are in central component.\n", vpts.size() );
#endif
        // These variables now hold avg hue, sat for the component.
        avghue/=pctr;
        avgsat/=pctr;
        
        // Calculate a rotated rectangle to fit the center component.
        RotatedRect rec = minAreaRect( vpts );
        // Draw the rotated rect around the center component.
        Point2f vertices[4];
        rec.points(vertices);
        for(int i = 0; i < 4; i++){
            line(frame, vertices[i], vertices[(i+1)%4], Scalar(0,255,0), 1, 4);
        }
        // Calculate the moments of the points in the center component.
        Moments mom = moments( vpts );
        // Get the 7 Hu moments.
        vector<double> hu;
        HuMoments( mom, hu );
        // add considerations for hue and saturation. The variable,
        // despite being called 'hu', also holds these values.
        hu.push_back( avghue );
        hu.push_back( avgsat );
#ifdef DEBUG
        printf("9 FV calculated.\n");
#endif
        
        // Call the function to query this object on the screen based on its FV.
        if ( entries > 0 ) {
            
            string match = queryFV( hu );
            string displaynamestr = ( "ID: " + match );
            // write the name of closest value from the FV DB query.
            putText(frame, displaynamestr, (vertices[2],vertices[(2+1)%4]), FONT_HERSHEY_PLAIN, 1,Scalar(255,255,255),1
                    );
        }
        
        
        // Draw the first Hu value on the frame.
        ostringstream sstream;
        sstream << hu[0];
        string hu0 = sstream.str();
        string displaystr = ( "Hu[0]: " + hu0 );
        putText(frame, displaystr, (vertices[0],vertices[(0+1)%4]), FONT_HERSHEY_PLAIN, 1, Scalar(255,255,255),1);
#ifdef DEBUG
        printf("After both putText calls.\n");
#endif
        imshow("normal", frame );
        imshow("thresh", dil );
        imshow("hsv", hsv );
#ifdef DEBUG
        printf("After both show frames\n");
#endif
        
        
        //        vector<Mat> mats;
        //        mats.push_back( frame );
        //        mats.push_back( dil );
        //        mats.push_back( hsv );
        //        Mat canv = makeCanvas( mats, 1000, 2 );
        
        
        // wait for the user to press a key
#ifdef DEBUG
        printf("Before waitkey\n");
#endif
        int keypress = waitKey(60);
        
        // if 'n', insert current FV to FV DB.
        if ( keypress == 110 ) {
            printf( "Enter into FV DB\n" );
            string oname = "";
            printf("\nEnter this object's name\n>");
            getline( cin, oname );
            insertFV( oname, hu );
            
            // if 'r', clear the FV DB.
        } else if (keypress == 114 ){
            cout << "Clearing data in FV DB.\n";
            ofstream insertfile;
            insertfile.open( "P3FeatureVectors.txt");
            insertfile.close();
            // if '1', change distance metric to Euclidean.
        } else if (keypress == 49 ){
            cout << "Distance Metric chosen: Euclidean\n";
            dmetric = "Euclidean";
            // if '2', change distance metric to KNN.
        } else if (keypress == 50 ){
            cout << "Distance Metric chosen: KNN\n";
            dmetric = "KNN";
            // if ESC then exit the program.
        } else if (keypress == 27 ){
            cout << "Exiting Object Recognition.\n";
            break;
        }
    }
    // terminate the video capture
    printf("Terminating\n");
    delete capdev;
    return(0);
}


/* function insertFV takes the name entered by the user and writes it to
 the end of already-in-existence "P3FeatureVectors.txt". */
void insertFV( string name, vector<double> features ){
    ofstream insertfile;
    insertfile.open( "P3FeatureVectors.txt", std::ios_base::app);
    if( not (insertfile.is_open())){
        printf( "error opening P3FeatureVectors.txt for insertion.\n" );
        return;
    }
    insertfile << name;
    insertfile << "\t";
    for( int i = 0; i < features.size(); i++ ){
        insertfile << features[i];
        insertfile << "\t";
    }
    insertfile << "\n";
    printf( "Entry named %s inserted to FV DB successfully.\n", name.c_str() );
    insertfile.close();
    entries++;
    return;
}

/* function queryFV takes in the features of the object currently on the
 screen and attempts to find the nearest object using the global distance
 metric. */
string queryFV( vector<double> features ){
    
    // It's hard-coded to find "P3FeatureVectors.txt".
    ifstream file;
    file.open("P3FeatureVectors.txt");
    if (not (file.is_open())){
        printf( "error opening P3FeatureVectors.txt for query.\n" );
        return "";
    }
    string line;
    
    // This is a map to keep names and vals neat and sorted.
    map<double,string> namedistance;
    int linecounter = 0;
    // Matrix so that we can keep track of all values across objects.
    vector< vector<double> >            valmatrix;
    valmatrix.resize( entries, vector<double>( 9 ) );
    vector< string > names;
    // This reads the 9 values into a matrix
    while( getline(file, line))
    {
        stringstream   linestream(line);
        string         name;
        
        linestream >> name;
        names.push_back(name);
        // Read the doubles using the operator >>
        for (int i = 0; i < 9; i++){
            double val;
            linestream >> val;
            valmatrix[linecounter][i] = val;
        }
        linecounter ++;
#ifdef DEBUG
        printf("while loop\n");
#endif
    }
    file.close();
    
#ifdef DEBUG
    cout << "LOOOPING THROUGH THE VALMATRIX\n\n\n";
    
    cout << "overall\n";
    cout << valmatrix.size();
    cout << "of one row\n";
    cout << valmatrix[0].size();
    cout << "\n";
    for( int i = 0; i < valmatrix.size(); i++ ){
        for ( int j = 0; j < valmatrix[i].size(); j++){
            cout << valmatrix[i][j];
            cout << ",";
        }
        cout << "~~";
    }
#endif
    
    
    // These are the calculations for scaled Euclidean distance
    if ( dmetric == "Euclidean" ){
        // These calculations are to get the standard deviation of each of the
        // 9 features for then calculating the scaled Euclidean distance.
        vector<double> stddevs( 9 );
        for ( int b = 0; b < 9; ++b ){
            double sum = 0;
            int ecop = entries;
            vector<double> diff( ecop );
            vector<double> vals( ecop );
            for ( int j = 0; j < entries; ++j ){
                double v = valmatrix[j][b];
                sum += v;
                vals.push_back( v );
#ifdef DEBUG
                printf("j loop euc\n");
#endif
            }
            double mean = sum/double(ecop);
            // transform(vals.begin(), vals.end(), diff.begin(), bind2nd(minus<double>(), mean));
            // double diff = 0;
            for( int l = 0; l < vals.size(); l++ ){
                diff.push_back( vals[l] - mean );
            }
            double sqsum = 0;
            for (int k = 0; k<ecop; k++) {
                sqsum += (diff[k] * diff[k]);
#ifdef DEBUG
                printf("k loop euc\n");
#endif
            }
            double stddev = sqrt( sqsum/ ecop );
            stddevs[b] =  stddev ;
            
#ifdef DEBUG
            printf("b loop euc\n");
#endif
        }
        
#ifdef DEBUG
        printf("between euc looops\n");
#endif
        
        for( int i = 0; i < entries; i++ ){
            double s_euc_dis = 0;
            for( int j = 0; j < 9; j++){
                
                if( stddevs[j] > 0.00001 ){
                    s_euc_dis += (features[j] - valmatrix[i][j])/(stddevs[j]);
                } else {
                    s_euc_dis += (features[j] - valmatrix[i][j])/(0.00001);
                }
#ifdef DEBUGeuc
                cout << "Features [j]\t";
                cout << (features[j] - valmatrix[i][j])/(0.001);
                cout << "\n\n\n";
                cout << "Distance:\t";
                cout << s_euc_dis;
                cout << "\t\tName:\t";
                cout << names[i];
                cout << "\n";
#endif
            }
            
            namedistance[abs(s_euc_dis)] = names[i];
#ifdef DEBUG
            printf("end second j loop euc\n");
            for(map<double,string>::reverse_iterator it = namedistance.rbegin();
                it != namedistance.rend(); ++it)
            {
                cout << it->first <<"\t";
                cout << it->second << "\t";
            }
            cout << "\n\n\n";
#endif
        }

        string retval = namedistance.begin()->second;
        return retval;
    }
    
    // These are the calculations for KNN distance metric.
    else if ( dmetric == "KNN" ){
        
        // Make a map to hold the classification string and the corresponding
        // integer.
        map<int, string> namemap;
        // the data needs to be in a matrix where successive entries are just
        // appended onto the end.
        Mat data(entries,9, CV_32F);
        Mat sampledata(1, 9, CV_32F);
        // fill a Mat for names ("classifications"), too.
        Mat classifications(0,entries,CV_32F);
        for (int i = 0; i < entries; i ++) {
            Mat tempdata(0,9,CV_32F);
            namemap[i] = names[i];
            classifications.push_back((float)i);
            for (int j = 0; j < 9; j++ ){
                if (i == 0){
                    sampledata.at<float>(i,j) =  ((float)features[j] );
                }
                data.at<float>(i,j) = ((float)valmatrix[i][j]);
            }
            //data.push_back(tempdata);
        }
#ifdef DEBUG
        cout << data;
        cout << "sampledatasize:\t\t" << sampledata.size() << "\n";
        cout << "datasize:\t\t" << data.size() << "\n";
        cout << "classifications size:\t\t" << classifications.size() << "\n";
        cout << "entries:\t\t" << entries << "\n";
        cout << "\n\n\n\n\n\n";
#endif
        Ptr<KNearest> knn = KNearest::create();
        Ptr<TrainData> trainingData = TrainData::create(data,
                                    SampleTypes::ROW_SAMPLE,classifications);
        knn->setIsClassifier(true);
        knn->setAlgorithmType(KNearest::Types::BRUTE_FORCE);
        knn->setDefaultK(1);
        
        knn->train(trainingData);
        Mat results(0,0,CV_32F);
        knn->findNearest(sampledata,knn->getDefaultK(),results);
        
        return ( namemap[results.at<float>(0,0)] );
    }
    return "";
}


int checkFVDB(){
    // It's hard-coded to find "P3FeatureVectors.txt".
    ifstream file;
    file.open("P3FeatureVectors.txt");
    // If the file is empty
    if (file.peek() == ifstream::traits_type::eof() ){
        // return a counter value of 0.
        printf( "The FV DB 'P3FeatureVectors.txt' contains 0 entries.\n");
        file.close();
        return 0;
        // if the file is not empty, total the number of entries.
    } else {
        string line;
        int entries_local = 0;
        while( getline(file, line)){
            entries_local++;
        }
        printf( "The FV DB 'P3FeatureVectors.txt' contains %d entries.\n", entries_local );
        file.close();
        return entries_local;
    }
    
}
