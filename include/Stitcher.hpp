// Include Guard
#ifndef __Stitcher_
#define __Stitcher_
//#pragma once	//don't need for pragma if include guard is set

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <opencv2/imgcodecs.hpp>		// imread
#include <opencv2/features2d.hpp>		// SIFT
#include <opencv2/calib3d.hpp>			// findhomography()
#include <opencv2/highgui.hpp>			// imshow
#include <opencv2/imgproc.hpp>			// warpperspective
#include <opencv2/imgproc/types_c.h>	// CV_BGR2GRAY
#include <opencv2/stitching.hpp>		// stitcher
#include <opencv2/core.hpp>				// addweighted()
#include <opencv2/core/types.hpp>		// Mat type (CV64FC1)
#include <opencv2/flann.hpp>			// hierarchicalClustering()
#include <Eigen/Dense>					// Include Eigen Matrices Manipulation
#include <iostream>						// cout
#include <string>						// string


using namespace std;
using namespace cv;
using namespace Eigen;

namespace A005 {
	
	class stitching_program
	{
	public:
		// Constructors
		// using default constructor
		// Deconstructor
		~stitching_program() {}

		// Stitching 2 frames
		// frame1 (frame n+1) // frame2 (frame n)
		static Mat stitch2frames(Mat frame1, Mat frame2);
		// For checking distribution of feature points to see if good match before stitching
		// frame1 (n+1) | frame2(n)
		static int Check_Points_Distribution(Mat frame1, Mat frame2);
		// Get current set Lowe's Ratio
		static float get_ratio();
		// changing lowe's ratio number
		static void change_ratio(int r);
		// getting arrangement of control pts
		static void get_controlpts();
		// changing arrangement/number of control pts
		// l_r : left row // l_c : left col // m_r : middle row // m_c : middle c
		// r_r : right row // r_c : right col
		static void change_controlpts(int l_r,int l_c, int m_r, int m_c, int r_r, int r_c);
		// get current set width allowance for stitched frame
		static int get_width_allowance();
		// Change width allowance for stitched frame
		static void change_width_allowance(int width);
		// Get current set percentage width of blend frame used for fixing pts
		static float get_perc_width_fixed();
		// Change percentage width of blend frame used for fixing pts
		static void change_perc_width_fixed(float perc);
		// get current set percentage width of n frame used for moving pts
		static float get_perc_width_moving();
		// change percentage width of n frame used for moving pts
		static void change_perc_width_moving(float perc);
		// removing blank portion right side of frame 
		static Mat remove_black_portion(Mat frame_uncut);
	private:
		static float ratio;
		static int left_row, left_col, middle_row, middle_col, right_row, right_col;
		static int width_allowance;
		static float perc_width_fixed;
		static float perc_width_moving;

		static int Min_Num_Clusters;
		static int Min_Num_MatchedFeaturePoints;
		static float Avg_Dist_Btw_Clusters;
	};


}


#endif