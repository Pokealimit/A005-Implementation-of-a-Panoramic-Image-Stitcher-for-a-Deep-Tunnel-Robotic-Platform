// ImageStitchingOpenCV.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
//#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include "LaplacianBlending.h"

using namespace cv;
using namespace std;
//using namespace cv::xfeatures2d;
int main()
{
    Mat img1,img2;
    img1 = imread("test1.jpg", IMREAD_COLOR); // Read the file
    img2 = imread("test2.jpg", IMREAD_COLOR); // Read the file

    //imshow("Display window", image); // Show our image inside it.
    //waitKey(0); // Wait for a keystroke in the window

    //For using SIFT feature
    cv::Ptr<cv::SIFT> siftPtr = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    siftPtr->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
    siftPtr->detectAndCompute(img2, noArray(), keypoints2, descriptors2);

    //-- Step 2: Matching descriptor vectors with a FLANN based matcher
    // Since SURF is a floating-point descriptor NORM_L2 is used
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<DMatch> > knn_matches;
    matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.75f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    //-- Draw matches
    Mat img_matches;
    drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, Scalar::all(-1),Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    //-- Show detected matches
    //namedWindow("matches", WINDOW_NORMAL); // create a window for display.
    //imshow("matches", img_matches);
    //waitKey(0);

    //-- Localize the object
    std::vector<Point2f> currframe;
    std::vector<Point2f> nextframe;
    for (size_t i = 0; i < good_matches.size(); i++)
    {
        //-- Get the keypoints from the good matches
        currframe.push_back(keypoints1[good_matches[i].queryIdx].pt);
        nextframe.push_back(keypoints2[good_matches[i].trainIdx].pt);
    }
    Mat H = findHomography(nextframe, currframe, RANSAC);
    Mat right;
    warpPerspective(img2, right, H, Size(img1.cols + img2.cols, img1.rows));
    namedWindow("RIGHT", WINDOW_NORMAL); // Create a window for display.
    imshow("RIGHT", right);
    waitKey(0);

    Mat I = Mat::eye(3, 3, CV_32F);
    Mat left;
    warpPerspective(img1, left, I, Size(img1.cols + img2.cols, img1.rows));
    //img1.copyTo(left(Rect(0, 0, img1.cols, img1.rows)));
    //result(0:img2.rows, 0 : img2.cols) = img2;
    namedWindow("LEFT", WINDOW_NORMAL); // Create a window for display.
    imshow("LEFT", left);
    waitKey(0);
    //Blending step
    Mat mask1, mask2, maskF;
    mask1 = (left > 0)/255;
    imshow("MASK1", mask1*255);
    waitKey(0);
    mask2 = (right > 0) / 255;
    imshow("MASK2", mask2*255);
    waitKey(0);
    maskF = (mask1 + mask2) > 1.5;
    imshow("MASKF",maskF*255);
    waitKey(0);
    cvtColor(maskF, maskF, cv::COLOR_RGB2GRAY);
    Mat thr;
    // convert grayscale to binary image
    threshold(maskF, thr, 100, 255, THRESH_BINARY);
    // find moments of the image
    Moments m = moments(thr, true);
    Point p(m.m10 / m.m00, m.m01 / m.m00);
    //cout << p << endl;
    //// show the image with a point mark at the centroid
    circle(maskF, p, 5, Scalar(128, 0, 0), -1);
    imshow("Image with center", maskF);
    waitKey(0);

    Mat_<float> cr_Mask(maskF.rows, maskF.cols, 0.0);
    cr_Mask(Range::all(), Range(0, p.x)) = 1.0;
    namedWindow("FINAL MASK", WINDOW_NORMAL);
    imshow("FINAL MASK", cr_Mask);
    //waitKey(0);

    //Mat l8u = imread("orange.jpg");
    //Mat r8u = imread("apple.jpg");


    Mat_<Vec3f> l; left.convertTo(l, CV_32F, 1.0 / 255.0);
    Mat_<Vec3f> r; right.convertTo(r, CV_32F, 1.0 / 255.0);
    //Mat_<float> mask(l.rows, l.cols, 0.0);
    //mask(Range::all(), Range(0, mask.cols / 2)) = 1.0;
    Mat_<Vec3f> blend = LaplacianBlend(l, r, cr_Mask);
    namedWindow("BLENDED", WINDOW_NORMAL);
    imshow("BLENDED", blend);
    waitKey(0);
    return 0;
}