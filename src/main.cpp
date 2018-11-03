
#include <fstream>
#include <iostream>
#include <filesystem>
#include <unistd.h>
#include <stdio.h>
#include <vector>
#include <iterator>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <regex>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/xfeatures2d/nonfree.hpp"

namespace fs = boost::filesystem;
namespace po = boost::program_options;
namespace bs = boost;



// int main(int argc, char* argv[])
int main(int argc, char* argv[])
{
  

  cv::Mat img_1 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
  // cv::resize(img_1, img_1, cv::Size(1080,720));


  
  
  if( !img_1.data )
  { std::cout<< " --(!) Error reading images " << std::endl; return -1; }


  int minHessian = 400;
  cv::FlannBasedMatcher matcher;
  cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create( minHessian );

  

  std::vector<cv::KeyPoint> keypoints_object;
  cv::Mat descriptors_object;
  detector->detectAndCompute(img_1, cv::Mat(), keypoints_object, descriptors_object);

  cv::imwrite("./detect.jpg", img_1);  
  

  cv::VideoCapture cap(0); // open the default camera
  if(!cap.isOpened())  // check if we succeeded
    return -1;
  cap.set(cv::CAP_PROP_FRAME_WIDTH, 840 );
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 680);

  cv::namedWindow("Capture",1);
  for(;;)
  {
    cv::Mat frame;
    cap >> frame; // get a new frame from camera

    cv::cvtColor(frame, frame, CV_BGR2GRAY);

    
    std::vector<cv::KeyPoint> keypoints_scene;
    cv::Mat descriptors_scene;;
    detector->detectAndCompute(frame, cv::Mat(), keypoints_scene, descriptors_scene );

    std::vector<cv::DMatch > matches;
    matcher.match( descriptors_object, descriptors_scene, matches );
    double max_dist = 0; double min_dist = 100;
    for( int i = 0; i < descriptors_object.rows; i++ )
    { double dist = matches[i].distance;
      if( dist < min_dist ) min_dist = dist;
      if( dist > max_dist ) max_dist = dist;
    }
    std::vector<cv::DMatch > good_matches;
    for( int i = 0; i < descriptors_object.rows; i++ )
    { if( matches[i].distance < 3*min_dist )
      {
        good_matches.push_back( matches[i]);
      }
    }

    std::vector<cv::Point2f> obj;
    std::vector<cv::Point2f> scene;
    for( size_t i = 0; i < good_matches.size(); i++ )
    {
      obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
      scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
    }
    cv::Mat H = cv::findHomography( obj, scene, cv::RANSAC );
    std::vector<cv::Point2f> obj_corners(4);
    obj_corners[0] = cvPoint(0,0);
    obj_corners[1] = cvPoint( img_1.cols, 0 );
    obj_corners[2] = cvPoint( img_1.cols, img_1.rows );
    obj_corners[3] = cvPoint( 0, img_1.rows );
    std::vector<cv::Point2f> scene_corners(4);
    perspectiveTransform( obj_corners, scene_corners, H);

    line( frame, scene_corners[0], scene_corners[1], cv::Scalar(0, 255, 0), 4 );
    line( frame, scene_corners[1], scene_corners[2], cv::Scalar( 0, 255, 0), 4 );
    line( frame, scene_corners[2], scene_corners[3], cv::Scalar( 0, 255, 0), 4 );
    line( frame, scene_corners[3], scene_corners[0], cv::Scalar( 0, 255, 0), 4 );

    
    
    imshow("Capture", frame);
    if(cv::waitKey(10) == 27 ) break;
  }

  

  
  
}
