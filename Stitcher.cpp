#include "include/Stitcher.hpp"
#include "include/LaplacianBlending.hpp"

float square(float x) { return x * x;}
float cube(float x) {return x * x * x;}
float quarter(float x) {return x * x * x * x;}
float power5(float x) {return x * x * x * x * x;}
float power6(float x) {return x * x * x * x * x * x;}
float power7(float x) { return x * x * x * x * x * x * x; }
float power8(float x) { return x * x * x * x * x * x * x * x; }
float power9(float x) { return x * x * x * x * x * x * x * x * x; }

namespace A005 {

	Mat stitching_program::stitch2frames(Mat frame1, Mat frame2, bool use_AKAZE) {
		Mat frame1_gray, frame2_gray;																// Storing frames
		vector<Point2f> edge_points, obj, scene;													// Store points
		vector <KeyPoint> keypoints1, keypoints2;													// Storing keypoints
		Mat des1, des2,result;																		// Storing Descriptors and results (some of the mat not in used - copied over from another stitching sln)
		Ptr<SIFT> detector = SIFT::create();														// Initialise Sift Detector
		Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);	// Initialise FLANN Matcher
		vector<vector<DMatch>> matches;																// Store matches
		vector<DMatch> good_matches;																// Store good matches

		// convert to grayscale for keypoints detection
		cvtColor(frame1, frame1_gray, CV_BGR2GRAY);
		cvtColor(frame2, frame2_gray, CV_BGR2GRAY);

		edge_points.push_back(Point2f(0, 0));
		edge_points.push_back(Point2f(frame1.cols,0));
		edge_points.push_back(Point2f(0, frame1.rows));
		edge_points.push_back(Point2f(frame1.cols,frame1.rows));

		// create a mask for ROI of feature detector - as there is no need to find keypoints of previous part of stitched image
		// Mat mask = Mat::zeros(Size(frame2.cols, frame2.rows), CV_8UC1);
		// rectangle(mask, cvPoint(frame2.cols - frame1.cols, 0), cvPoint(frame2.cols, frame2.rows), 255, -1);
		// Compute feature points and descriptors
		detector->detectAndCompute(frame1_gray, noArray(), keypoints1, des1);
		detector->detectAndCompute(frame2_gray, noArray(), keypoints2, des2);
		// Match feature points
		matcher->knnMatch(des1, des2, matches, 2);
		// only get good matching points using Lowe's ratio test
		for (int i = 0; i < matches.size(); ++i)
		{
			if (matches[i][0].distance < ratio * matches[i][1].distance)
			{
				good_matches.push_back(matches[i][0]);
			}
		}
		// Draw lines of good matches
		// drawMatches(frame1, keypoints1, frame2, keypoints2, good_matches, result, Scalar(0, 0, 255), Scalar(0, 0, 255));

		// -- Get the keypoints from the good matches
		for (size_t i = 0; i < good_matches.size(); i++){
			obj.push_back(keypoints1[good_matches[i].queryIdx].pt);
			scene.push_back(keypoints2[good_matches[i].trainIdx].pt);
		}


		// find homography
		Mat H = findHomography(obj,scene,RANSAC);
		cout << "Homography H : \n" << H << endl;
		// Warp frame1 to same location as frame2
		Mat right;
		warpPerspective(frame1,right,H,Size(frame1.cols + frame2.cols, frame1.rows));
		// Blending steps using LaplacianBlending
		Mat left = Mat::zeros(Size(frame1.cols+frame2.cols,frame2.rows),frame2.type());
		frame2.copyTo(left(Rect(0,0,frame2.cols,frame2.rows)));
		Mat mask1,mask2,maskF;
		mask1 = (left > 0)/255;
    	// imshow("MASK1", mask1*255);
    	// waitKey(0);
    	mask2 = (right > 0) / 255;
    	// imshow("MASK2", mask2*255);
    	// waitKey(0);
    	maskF = (mask1 + mask2) > 1.5;
    	// imshow("MASKF",maskF*255);
    	// waitKey(0);
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
    	// imshow("Image with center", maskF);
    	// waitKey(0);		
		Mat_<float> cr_Mask(maskF.rows, maskF.cols, 0.0);
    	cr_Mask(Range::all(), Range(0, p.x)) = 1.0;
    	// namedWindow("FINAL MASK", WINDOW_NORMAL);
    	// imshow("FINAL MASK", cr_Mask);
    	//waitKey(0);
    	Mat_<Vec3f> l; left.convertTo(l, CV_32F, 1.0 / 255.0);
    	Mat_<Vec3f> r; right.convertTo(r, CV_32F, 1.0 / 255.0);
    	//Mat_<float> mask(l.rows, l.cols, 0.0);
    	//mask(Range::all(), Range(0, mask.cols / 2)) = 1.0;
    	Mat_<Vec3f> blend = LaplacianBlend(l, r, cr_Mask);
    	// namedWindow("BLENDED", WINDOW_NORMAL);
    	// imshow("BLENDED", blend);
    	// waitKey(0);
		// result = blend.clone();

		//removing black part
		Mat blend_crop;
		perspectiveTransform(edge_points,edge_points,H);
		int cutoff = (edge_points[1].x > edge_points[3].x) ? edge_points[1].x : edge_points[3].x;
		Rect crop(0,0,cutoff,blend.rows);
		blend(crop).convertTo(blend_crop,CV_8U,255);	// convert 32bit floating point back to uint8 for display

		// Creating Points for Blended frame
		vector <Point2f> blend_frame_pts;
		float x_width = blend_crop.cols / (left_col+middle_col+right_col-1);
		float y_width = blend_crop.rows / (left_row-1);
		for (int i =0; i<left_row; i++){
			for(int j=0; j<left_col; j++){
				Point2f pt( j*x_width , i*y_width);
		 		blend_frame_pts.push_back(pt);
			}
		}
		y_width = blend_crop.rows/(middle_row-1);
		for (int i =0; i<middle_row; i++){
			for(int j=0; j<middle_col; j++){
				Point2f pt( (j+left_col)*x_width , i*y_width);
		 		blend_frame_pts.push_back(pt);
			}
		}
		y_width = blend_crop.rows/(right_row-1);
		for (int i =0; i<right_row; i++){
			for(int j=0; j<right_col; j++){
				Point2f pt( (j+left_col+middle_col)*x_width , i*y_width);
		 		blend_frame_pts.push_back(pt);
			}
		}
		Mat blend_frame_drawing = blend_crop.clone();
		// for drawing blended frame pts
		for(int i=0; i < blend_frame_pts.size(); i++)
			circle(blend_frame_drawing,blend_frame_pts[i],1,Scalar(0,255,0),5,8,0);		
		// imshow("blend frame pts",blend_frame_drawing); waitKey(0);

		// Creating pts for frame before stitching
		vector <Point2f> stitch_frame_pts;
		x_width = frame1.cols / (left_col+middle_col+right_col-1);
		y_width = frame1.rows / (left_row-1);
		for (int i =0; i<left_row; i++){
			for(int j=0; j<left_col; j++){
				Point2f pt( j*x_width , i*y_width);
		 		stitch_frame_pts.push_back(pt);
			}
		}
		y_width = frame1.rows/(middle_row-1);
		for (int i =0; i<middle_row; i++){
			for(int j=0; j<middle_col; j++){
				Point2f pt( (j+left_col)*x_width , i*y_width);
		 		stitch_frame_pts.push_back(pt);
			}
		}
		y_width = frame1.rows/(right_row-1);
		for (int i =0; i<right_row; i++){
			for(int j=0; j<right_col; j++){
				Point2f pt( (j+left_col+middle_col)*x_width , i*y_width);
		 		stitch_frame_pts.push_back(pt);
			}
		}
		Mat frame1_drawing = frame1.clone();
		// for drawing frame1 pts
		for(int i=0; i < stitch_frame_pts.size(); i++)
			circle(frame1_drawing,stitch_frame_pts[i],1,Scalar(0,255,0),5,8,0);		
		// imshow("frame1 pts",frame1_drawing); waitKey(0);

		// Splitting pts to be used for mapping
		vector <Point2f> left_edge_pts, right_edge_pts;
		for(int i=0; i<blend_frame_pts.size();i++){
			if(blend_frame_pts[i].x < (blend_crop.cols*perc_width_fixed)){
				left_edge_pts.push_back(blend_frame_pts[i]);
			}
		}
		for(int i=0; i<stitch_frame_pts.size();i++){
			if(stitch_frame_pts[i].x > (1-perc_width_moving)*frame1.cols){
				right_edge_pts.push_back(stitch_frame_pts[i]);
			}
		}
		// for drawing left edge pts
		Mat drawing_blend_w_pts = blend_crop.clone();
		for(int i=0; i < left_edge_pts.size(); i++)
			circle(drawing_blend_w_pts,left_edge_pts[i],1,Scalar(0,255,0),5,8,0);		
		// imshow("left edge pts",drawing_blend_w_pts); waitKey(0);
		// for drawing right edge pts
		Mat temp = frame1.clone();
		for(int i=0; i < right_edge_pts.size(); i++)
			circle(temp,right_edge_pts[i],1,Scalar(0,255,0),5,8,0);		
		// imshow("right edge pts",temp); waitKey(0);
		// create right edge points wrt to blend frame (x translation ONLY)
		vector <Point2f> right_edge_pts_t;
		double x_trans;
		x_trans = (H.at<double>(0,2) > H.at<double>(2,0)) ? H.at<double>(0,2) : H.at<double>(2,0);
		for(int i=0; i<right_edge_pts.size();i++){
			Point2f pt( right_edge_pts[i].x + x_trans , right_edge_pts[i].y);
			right_edge_pts_t.push_back(pt);
		}
		// transform right edge pts to blend frame
		perspectiveTransform(right_edge_pts,right_edge_pts,H);
		// draw right edge pts on blend frame with left edge pts
		for(int i=0; i < right_edge_pts.size(); i++){
			circle(drawing_blend_w_pts,right_edge_pts[i],1,Scalar(0,255,0),5,8,0);
			circle(drawing_blend_w_pts,right_edge_pts_t[i],1,Scalar(0,0,255),5,8,0);
		}
		// imshow("left and right edge pts",drawing_blend_w_pts); waitKey(0);

		// Declaring Matrix for solving coefficient of polynomial transform
		Matrix<float,Dynamic,Dynamic> Corner_pts(left_edge_pts.size()+right_edge_pts.size(), 2);
		Matrix<float,Dynamic,Dynamic> Template_pts(left_edge_pts.size()+right_edge_pts_t.size(), 2);
		Matrix<float,Dynamic,Dynamic> x(left_edge_pts.size()+right_edge_pts.size(), 1);
		Matrix<float,Dynamic,Dynamic> y(left_edge_pts.size()+right_edge_pts.size(), 1);
		Matrix<float,Dynamic,Dynamic> xp(left_edge_pts.size()+right_edge_pts.size(), 1);
		Matrix<float,Dynamic,Dynamic> yp(left_edge_pts.size()+right_edge_pts.size(), 1);
		Matrix<float,Dynamic,Dynamic> A(left_edge_pts.size()+right_edge_pts.size(), 28);	// sixth order polynomial
		Matrix<float,28,1> A0,B0;															// sixth order polynomial

		// Inserting control pts & moving pts
		for( int i = 0 ; i < left_edge_pts.size() ; i++ ){
			for(int j = 0 ; j < 2 ; j++ ){
				if (j == 0)
					Template_pts(i,j) = Corner_pts(i,j) = left_edge_pts[i].x;
				else
					Template_pts(i,j) = Corner_pts(i,j) = left_edge_pts[i].y;
			}	
		}
		for( int i = 0 ; i < right_edge_pts_t.size() ; i++ ){
			for(int j = 0 ; j < 2 ; j++ ){
				if (j == 0){
					Template_pts(i+left_edge_pts.size(),j) = right_edge_pts_t[i].x;
					Corner_pts(i+left_edge_pts.size(),j) = right_edge_pts[i].x;
				}
				else{
					Template_pts(i+left_edge_pts.size(),j) = right_edge_pts_t[i].y;
					Corner_pts(i+left_edge_pts.size(),j) = right_edge_pts[i].y;
				}
			}	
		}

		// for drawing circle of 24 warpped pts
		Mat compare_template_corner = blend_crop.clone();
		for (int i = 0; i < (left_edge_pts.size()+right_edge_pts.size()); i++) {
			circle(compare_template_corner, cvPoint(Template_pts(i, 0), Template_pts(i, 1)), 1, Scalar(0, 0, 255), 5, 8, 0);
			circle(compare_template_corner, cvPoint(Corner_pts(i, 0), Corner_pts(i, 1)), 1, Scalar(0, 255, 0), 5, 8, 0);
		}
		// imshow("template vs corner",compare_template_corner); waitKey(0);
		// imwrite("compare_temp_corner_beforewarp.jpg",compare_template_corner);

		// Splitting points into x and y component for solving
		for (int i = 0; i < (left_edge_pts.size()+right_edge_pts.size()); i++) {
			x(i) = Template_pts(i, 0);
			y(i) = Template_pts(i, 1);
			xp(i) = Corner_pts(i, 0);
			yp(i) = Corner_pts(i, 1);
		}
		
		// Sixth order polynomail
		for (int i = 0; i < (left_edge_pts.size()+right_edge_pts.size()); i++) {
			A(i, 0) = 1;
			A(i, 1) = x(i);
			A(i, 2) = y(i);
			A(i, 3) = x(i) * y(i);
			A(i, 4) = square(x(i));
			A(i, 5) = square(y(i));
			A(i, 6) = square(x(i)) * y(i);
			A(i, 7) = x(i) * square(y(i));
			A(i, 8) = cube(x(i));
			A(i, 9) = cube(y(i));
			A(i, 10) = cube(x(i)) * y(i);
			A(i, 11) = square(x(i)) * square(y(i));
			A(i, 12) = x(i) * cube(y(i));
			A(i, 13) = quarter(x(i));
			A(i, 14) = quarter(y(i));
			A(i, 15) = quarter(x(i)) * y(i);
			A(i, 16) = cube(x(i)) * square(y(i));
			A(i, 17) = square(x(i)) * cube(y(i));
			A(i, 18) = x(i) * quarter(y(i));
			A(i, 19) = power5(x(i));
			A(i, 20) = power5(y(i));
			A(i, 21) = power5(x(i)) * y(i);
			A(i, 22) = quarter(x(i)) * square(y(i));
			A(i, 23) = cube(x(i)) * cube(y(i));
			A(i, 24) = square(x(i)) * quarter(y(i));
			A(i, 25) = x(i) * power5(y(i));
			A(i, 26) = power6(x(i));
			A(i, 27) = power6(y(i));
		}

		// Solving for coefficients
		A0 = A.householderQr().solve(xp);
		B0 = A.householderQr().solve(yp);

		// cout << "A : \n" << A << endl;
		// cout << "A0 : \n" << A0 << endl;
		// cout << "B0 : \n" << B0 << endl;
		double relative_error_A = (A * A0 - xp).norm() / xp.norm();
		double relative_error_B = (A * B0 - yp).norm() / yp.norm();
		double relative_error = relative_error_A + relative_error_B;
		// cout << "The relative error A is: " << relative_error_A << endl;
		// cout << "The relative error B is: " << relative_error_B << endl;
		cout << "The relative error Total is: " << relative_error << endl;
		// if(isnan(relative_error_A) || isnan(relative_error_B) || isnan(relative_error)) return Mat{};
		if(isnan(relative_error)) return Mat{};

		// creating xmap and ymap (Sixth Order Polynomial)
		Mat_<float> map_x(blend_crop.size()), map_y(blend_crop.size());
		for (int i = 0; i < map_x.rows; i++)
		{
			for (int j = 0; j < map_x.cols; j++)
			{
				map_x(i, j) = A0(0) + A0(1) * j + A0(2) * i + A0(3) * j * i + A0(4) * j * j + A0(5) * i * i + A0(6) * i * j * j + A0(7) * i * i * j + A0(8) * j * j * j + A0(9) * i * i * i + A0(10) * j * j * j * i + A0(11) * i * i * j * j + A0(12) * i * i * i * j
					+ A0(13) * j * j * j * j + A0(14) * i * i * i * i + A0(15) * j * j * j * j * i + A0(16) * j * j * j * i * i + A0(17) * j * j * i * i * i + A0(18) * j * i * i * i * i + A0(19) * j * j * j * j * j + A0(20) * i * i * i * i * i + A0(21) * j * j * j * j * j * i
					+ A0(22) * j * j * j * j * i * i + A0(23) * j * j * j * i * i * i + A0(24) * j * j * i * i * i * i + A0(25) * j * i * i * i * i * i + A0(26) * j * j * j * j * j * j + A0(27) * i * i * i * i * i * i;
				map_y(i, j) = B0(0) + B0(1) * j + B0(2) * i + B0(3) * j * i + B0(4) * j * j + B0(5) * i * i + B0(6) * i * j * j + B0(7) * i * i * j + B0(8) * j * j * j + B0(9) * i * i * i + B0(10) * j * j * j * i + B0(11) * i * i * j * j + B0(12) * i * i * i * j
					+ B0(13) * j * j * j * j + B0(14) * i * i * i * i + B0(15) * j * j * j * j * i + B0(16) * j * j * j * i * i + B0(17) * j * j * i * i * i + B0(18) * j * i * i * i * i + B0(19) * j * j * j * j * j + B0(20) * i * i * i * i * i + B0(21) * j * j * j * j * j * i
					+ B0(22) * j * j * j * j * i * i + B0(23) * j * j * j * i * i * i + B0(24) * j * j * i * i * i * i + B0(25) * j * i * i * i * i * i + B0(26) * j * j * j * j * j * j + B0(27) * i * i * i * i * i * i;
			}
		}

		// for drawing comparison after warpping
		Mat compare_template_corner_w = blend_crop.clone();
		for (int i = 0; i < (left_edge_pts.size()+right_edge_pts.size()); i++) 
			circle(compare_template_corner_w, cvPoint(Corner_pts(i, 0), Corner_pts(i, 1)), 1, Scalar(0, 255, 0), 5, 8, 0);

		remap(blend_crop, result, map_x, map_y, INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
		// remap(compare_template_corner_w, compare_template_corner_w, map_x, map_y, INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
		// for (int i = 0; i < (left_edge_pts.size()+right_edge_pts.size()); i++) 
		// 	circle(compare_template_corner_w, cvPoint(Template_pts(i, 0), Template_pts(i, 1)), 1, Scalar(0, 0, 255), 5, 8, 0);

		// imshow("compare template and corner warpped",compare_template_corner_w); waitKey(0);
		// imwrite("compare_temp_corner_afterwarp.jpg",compare_template_corner_w);

		// Checking absolute difference btw blended and warped frame
		// cout << "blend frame size : " << endl << blend_crop.size() << endl;
		// cout << "warpped frame size : " << endl << compare_template_corner_w.size() << endl;
		// Mat blend_crop_grey, result_grey, diff_btw_frames;
		// cvtColor(blend_crop, blend_crop_grey, CV_BGR2GRAY);
		// cvtColor(result, result_grey, CV_BGR2GRAY);
		// absdiff(blend_crop_grey,result_grey,diff_btw_frames);
		// imshow("absolute difference",diff_btw_frames); waitKey(0);
		// imwrite("./result/absdiff_frame0-5.jpg",diff_btw_frames);

		return result;
	}

	Mat stitching_program::remove_black_portion(Mat frame_uncut){

		// using cv::reduce to crop out black area
		Mat out;
		int cutOffIdx = 0;
		reduce(frame_uncut,out,0,REDUCE_MAX);
		for (int col = out.cols - 1; col > 0; --col) {
   			const Vec3b& vec = out.at<Vec3b>(0, col);
    		if (vec[0] || vec[1] || vec[2]) {
        		cutOffIdx = col;
        		break;
    		}
		}
		cout << "coutoffIdx :\t" << cutOffIdx << endl;
		Mat result = frame_uncut(Rect(0,0,cutOffIdx,frame_uncut.rows));
		return result;
	}

	int stitching_program::Check_Points_Distribution(Mat frame1, Mat frame2, bool use_AKAZE){
		Mat frame1_gray, frame2_gray;																// Storing frames
		vector<Point2f> edge_points, obj, scene;													// Store points
		vector <KeyPoint> keypoints1, keypoints2;													// Storing keypoints
		Mat des1, des2,result;																		// Storing Descriptors and results
		Ptr<SIFT> detector = SIFT::create();														// Initialise Sift Detector
		Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);	// Initialise FLANN Matcher
		vector<vector<DMatch>> matches;																// Store matches
		vector<DMatch> good_matches;																// Store good matches

		// convert to grayscale for keypoints detection
		cvtColor(frame1, frame1_gray, CV_BGR2GRAY);
		cvtColor(frame2, frame2_gray, CV_BGR2GRAY);

		// Compute feature points and descriptors
		detector->detectAndCompute(frame1_gray, noArray(), keypoints1, des1);
		detector->detectAndCompute(frame2_gray, noArray(), keypoints2, des2);
		// Match feature points
		matcher->knnMatch(des1, des2, matches, 2);
		// only get good matching points using Lowe's ratio test
		for (int i = 0; i < matches.size(); ++i)
		{
			if (matches[i][0].distance < ratio * matches[i][1].distance)
			{
				good_matches.push_back(matches[i][0]);
			}
		}
		// Draw lines of good matches
		drawMatches(frame1, keypoints1, frame2, keypoints2, good_matches, result, Scalar(0, 0, 255), Scalar(0, 0, 255)); imshow("SIFT Matches",result); waitKey(0);
		// -- Get the keypoints from the good matches
		for (size_t i = 0; i < good_matches.size(); i++){
			obj.push_back(keypoints1[good_matches[i].queryIdx].pt);
			scene.push_back(keypoints2[good_matches[i].trainIdx].pt);
		}

		/* AKAZE Feature Detector */
		if(use_AKAZE){
			vector<KeyPoint> AKAZE_keypoints_1, AKAZE_keypoints_2;
			Mat AKAZE_descriptors_1, AKAZE_descriptors_2;
			vector<vector<DMatch> > AKAZE_matches;
			// vector<DMatch> ORB_matches;
			vector<DMatch> AKAZE_good_matches;
			Ptr<AKAZE> AKAZE_detector = AKAZE::create();
			// Ptr<DescriptorMatcher> AKAZE_matcher  = DescriptorMatcher::create ( "" );
			BFMatcher AKAZE_matcher(NORM_HAMMING);
			// cout << "Creating ORB" << endl;
			// Ptr<DescriptorMatcher> ORB_matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
			AKAZE_detector -> detectAndCompute(frame1_gray, noArray(), AKAZE_keypoints_1, AKAZE_descriptors_1);
			AKAZE_detector -> detectAndCompute(frame2_gray, noArray(), AKAZE_keypoints_2, AKAZE_descriptors_2);
			// cout << "Detect and compute..." << endl;
			// ORB_matcher -> match ( ORB_descriptors_1, ORB_descriptors_2, ORB_matches );
			AKAZE_matcher.knnMatch(AKAZE_descriptors_1, AKAZE_descriptors_2, AKAZE_matches, 2);
			// cout<< "matching..." << endl;
			// only get good matching points using Lowe's ratio test
			for (int i = 0; i < AKAZE_matches.size(); ++i)
			{
				if (AKAZE_matches[i][0].distance < ratio * AKAZE_matches[i][1].distance)
					AKAZE_good_matches.push_back(AKAZE_matches[i][0]);
				// if (ORB_matches[i].distance < ratio * ORB_matches[i].distance)
				// 	ORB_good_matches.push_back(ORB_matches[i]);
			}
			drawMatches(frame1, AKAZE_keypoints_1, frame2, AKAZE_keypoints_2, AKAZE_good_matches, result, Scalar(0, 0, 255), Scalar(0, 0, 255)); imshow("AKAZE matches",result); waitKey(0);
			// cout << "found good matches..." << endl;
			vector<Point2f> AKAZE_obj, AKAZE_scene;
			for (size_t i = 0; i < AKAZE_good_matches.size(); i++){
				obj.push_back(AKAZE_keypoints_1[AKAZE_good_matches[i].queryIdx].pt);
				scene.push_back(AKAZE_keypoints_2[AKAZE_good_matches[i].trainIdx].pt);
			}

			/* Checking for duplicate points */
			vector <Point2f> obj_unique, scene_unique;
			for (size_t i=0; i<scene.size(); i++){
				for (size_t j=i+1; j <= scene.size(); j++){
					if( scene[i] == scene[j]) break;
					else if( j== scene.size() ){
						scene_unique.push_back(scene[i]);
						obj_unique.push_back(obj[i]);
					}
				}
			}
			cout << "Using AKAZE with SIFT... unique vs non-unique pts:\t" << obj_unique.size() << " vs " << obj.size() << endl;

			vector<DMatch> matches(scene_unique.size());
			for(size_t i = 0; i <scene_unique.size(); ++i)
  				matches[i] = cv::DMatch(i, i, 0);
			// make keypoints 
			vector<KeyPoint> kp_left(scene_unique.size());
			for(size_t i = 0; i < scene_unique.size(); ++i)
				kp_left[i] = cv::KeyPoint(scene_unique[i], 1);

			vector<KeyPoint> kp_right(obj_unique.size());
			for(size_t i = 0; i < obj_unique.size(); ++i)
				kp_right[i] = cv::KeyPoint(obj_unique[i], 1);
			
			Mat matches_drawing;
			drawMatches(frame2, kp_left, frame1, kp_right, matches, matches_drawing, Scalar(0, 0, 255), Scalar(0, 0, 255)); imshow("All Matches",matches_drawing); waitKey(0);


			/* Findind translation of f1 to find overlap area */
			Mat H = findHomography(obj_unique,scene_unique,RANSAC);
			double x_trans = (H.at<double>(0,2) > H.at<double>(2,0)) ? H.at<double>(0,2) : H.at<double>(2,0);
			cout << "x_trans = " << x_trans << endl;
			double overlap_area = (frame2.cols - x_trans) * frame2.rows;
			
			/* K Means Clustering */
			Mat labels, centers;
			int K=2, attempts=10, flags=KMEANS_RANDOM_CENTERS;
			TermCriteria tc;
			kmeans(obj_unique,K,labels,tc,attempts,flags,centers);
			Mat centers_points = centers.reshape(2,centers.rows);

			Scalar colorTab[] =
			{
			Scalar(0, 0, 255),
			Scalar(0,255,0),
			Scalar(255,100,100),
			Scalar(255,0,255),
			Scalar(0,255,255)
			};

			Mat drawing;
			drawing = frame1.clone();
			vector<Point2f> contour0, contour1;	
			
			cout << "Centers Points :\n" << centers_points << endl;

			// for drawing cluster center points
			for(int i=0; i < centers_points.rows; i++)
				circle(drawing,centers_points.at<Point2f>(i),1,colorTab[2],5,8,0);
			
			// cout << "labels :" << endl << labels << endl;

			// for drawing good matching points
			for( int i = 0; i < obj_unique.size(); i++ )
			{
				int clusterIdx = labels.at<int>(i);
				// circle( drawing, obj[i], 2, colorTab[clusterIdx], FILLED, LINE_AA );
				circle(drawing,obj_unique[i],1,colorTab[clusterIdx],5,8,0);

				if(clusterIdx) contour1.push_back(obj_unique[i]);
				else contour0.push_back(obj_unique[i]);
			}

			// for calculating area of clusters
			double area0 = contourArea(contour0);
			double area1 = contourArea(contour1);
			vector<vector<Point> > contours;
			vector<Point> approx0, approx1;
			approxPolyDP(contour0, approx0, 5, true);
			approxPolyDP(contour1, approx1, 5, true);
			double approx0_area = contourArea(approx0);
			double approx1_area = contourArea(approx1);

			contours.push_back(approx0);
			contours.push_back(approx1);

			cout << "area0 = " << area0 << endl << "approx0_area =" << approx0_area << endl; //<< "approx poly vertices" << approx0.size() << endl;
			cout << "area1 = " << area1 << endl << "approx1_area =" << approx1_area << endl; //<< "approx poly vertices" << approx1.size() << endl;
			cout << "overlapped area = " << overlap_area << endl;
			cout << "percentage area (approx) = " << ( (approx0_area + approx1_area) / overlap_area ) * 100 << " %" << endl;

			drawContours(drawing,contours,0,colorTab[3],2);
			drawContours(drawing,contours,1,colorTab[3],2);
			// imshow("Area covered by approx",drawing); waitKey(0);

			vector<vector<Point> > hull(contours.size());
			for(size_t i=0; i<contours.size(); i++)
				convexHull(contours[i],hull[i]);

			drawContours(drawing,hull,0,colorTab[4],2);
			drawContours(drawing,hull,1,colorTab[4],2);

			double hull0_area = contourArea(hull[0]);
			double hull1_area = contourArea(hull[1]);
			cout << "hull0_area =" << hull0_area << endl;
			cout << "hull1_area =" << hull1_area << endl;
			cout << "percentage area (hull) = " << ( (hull0_area + hull1_area) / overlap_area ) * 100 << " %" << endl;
			cout << "no. of matched points in C1=\t" << contour0.size() << endl;
			cout << "no. of matched points in C2=\t" << contour1.size() << endl;

			imshow("Actual area covered vs approx",drawing); waitKey(0);
			// return 1;
			// return drawing;
			if( (hull0_area + hull1_area) / overlap_area > 0.5 && contour0.size() > 40 && contour1.size() > 40 ) return 1;
			else return 0;
		}

		else{
			cout << "Not using AKAZE with SIFT... No of points:\t" << obj.size() << endl;
			/* Findind translation of f1 to find overlap area */
			Mat H = findHomography(obj,scene,RANSAC);
			double x_trans = (H.at<double>(0,2) > H.at<double>(2,0)) ? H.at<double>(0,2) : H.at<double>(2,0);
			cout << "x_trans = " << x_trans << endl;
			double overlap_area = (frame2.cols - x_trans) * frame2.rows;
			
			/* K Means Clustering */
			Mat labels, centers;
			int K=2, attempts=10, flags=KMEANS_RANDOM_CENTERS;
			TermCriteria tc;
			kmeans(obj,K,labels,tc,attempts,flags,centers);
			Mat centers_points = centers.reshape(2,centers.rows);

			Scalar colorTab[] =
			{
			Scalar(0, 0, 255),
			Scalar(0,255,0),
			Scalar(255,100,100),
			Scalar(255,0,255),
			Scalar(0,255,255)
			};

			Mat drawing;
			drawing = frame1.clone();
			vector<Point2f> contour0, contour1;	
			
			cout << "Centers Points :\n" << centers_points << endl;

			// for drawing cluster center points
			for(int i=0; i < centers_points.rows; i++)
				circle(drawing,centers_points.at<Point2f>(i),1,colorTab[2],5,8,0);
			
			// cout << "labels :" << endl << labels << endl;

			// for drawing good matching points
			for( int i = 0; i < obj.size(); i++ )
			{
				int clusterIdx = labels.at<int>(i);
				// circle( drawing, obj[i], 2, colorTab[clusterIdx], FILLED, LINE_AA );
				circle(drawing,obj[i],1,colorTab[clusterIdx],5,8,0);

				if(clusterIdx) contour1.push_back(obj[i]);
				else contour0.push_back(obj[i]);
			}

			// for calculating area of clusters
			double area0 = contourArea(contour0);
			double area1 = contourArea(contour1);
			vector<vector<Point> > contours;
			vector<Point> approx0, approx1;
			approxPolyDP(contour0, approx0, 5, true);
			approxPolyDP(contour1, approx1, 5, true);
			double approx0_area = contourArea(approx0);
			double approx1_area = contourArea(approx1);

			contours.push_back(approx0);
			contours.push_back(approx1);

			cout << "area0 = " << area0 << endl << "approx0_area =" << approx0_area << endl; //<< "approx poly vertices" << approx0.size() << endl;
			cout << "area1 = " << area1 << endl << "approx1_area =" << approx1_area << endl; //<< "approx poly vertices" << approx1.size() << endl;
			cout << "overlapped area = " << overlap_area << endl;
			cout << "percentage area (approx) = " << ( (approx0_area + approx1_area) / overlap_area ) * 100 << " %" << endl;

			drawContours(drawing,contours,0,colorTab[3],2);
			drawContours(drawing,contours,1,colorTab[3],2);
			// imshow("Area covered by approx",drawing); waitKey(0);

			vector<vector<Point> > hull(contours.size());
			for(size_t i=0; i<contours.size(); i++)
				convexHull(contours[i],hull[i]);

			drawContours(drawing,hull,0,colorTab[4],2);
			drawContours(drawing,hull,1,colorTab[4],2);

			double hull0_area = contourArea(hull[0]);
			double hull1_area = contourArea(hull[1]);
			cout << "hull0_area =" << hull0_area << endl;
			cout << "hull1_area =" << hull1_area << endl;
			cout << "percentage area (hull) = " << ( (hull0_area + hull1_area) / overlap_area ) * 100 << " %" << endl;
			cout << "no. of matched points in C1=\t" << contour0.size() << endl;
			cout << "no. of matched points in C2=\t" << contour1.size() << endl;

			imshow("Actual area covered vs approx",drawing); waitKey(0);
			// return 1;
			// return drawing;
			if( (hull0_area + hull1_area) / overlap_area > 0.5 && contour0.size() > 40 && contour1.size() > 40 ) return 1;
			else return 0;

		}
		/* Hierachical Clustering */
		// cvflann::KMeansIndexParams kmean_params(32,100,cvflann::FLANN_CENTERS_KMEANSPP);
		// // cvflann::KMeansIndexParams kmean_params(32,100,cvflann::FLANN_CENTERS_RANDOM);
		// Mat1f samples(obj.size(),2);

		// for(int i=0;i<obj.size();i++){
		// 	samples(i,0) = obj[i].x;
		// 	samples(i,1) = obj[i].y;
		// }
		// Mat1f centers(obj.size(),2);
		// // int true_number_clusters = flann::hierarchicalClustering<cv::L2<float> >(samples,centers,kmean_params);
		// int true_number_clusters = flann::hierarchicalClustering<cvflann::L2<float>>(samples,centers,kmean_params);
		// // int true_number_clusters = flann::hierarchicalClustering(samples,centers,kmean_params,100);
		// centers = centers.rowRange(cv::Range(0,true_number_clusters));

		// Mat drawing;
		// drawing = frame1.clone();
		// // for drawing points matching points
		// for(int i=0; i < obj.size(); i++)
		// 	circle(drawing,obj[i],1,Scalar(0,255,0),5,8,0);		
		
		// cout << "Centers Points :\n" << centers << endl;
		// // for drawing cluster points
		// for(int i=0; i < centers.rows; i++)
		// 	circle(drawing,centers.at<Point2f>(i),3,Scalar(0,0,255),5,8,0);
		
		
		// cout << "Number of Cluster Points :\t" << centers.rows << endl;
		// cout << "Number of Matched Feature Points :\t" << obj.size() << endl;
		// // imshow("cluster points",drawing); waitKey(0);

		// if (centers.rows < Min_Num_Clusters || obj.size() < Min_Num_MatchedFeaturePoints)
		// 	return 0;

		// /* Calculating Average Distance Between Each Cluster Points */
		// float dist_btw_clusters = 0;
		// for(int i=0; i<centers.rows; i++){
		// 	float distance = 0;
		// 	for(int j = i; j<centers.rows; j++)
		// 		distance += norm(centers.at<Point2f>(j) - centers.at<Point2f>(j+1));
		// 	float avg_dist = distance / (centers.rows - i);
		// 	dist_btw_clusters += avg_dist;
		// }
		// float avg_dist_btw_clusters = dist_btw_clusters / centers.rows;
		// cout << "Average Dist. Btw Clusters :\t" << avg_dist_btw_clusters << endl;

		// if (avg_dist_btw_clusters < Avg_Dist_Btw_Clusters)
		// 	return 0;

		// return 1;

		/* ORB Feature Detector */
		// vector<KeyPoint> ORB_keypoints_1, ORB_keypoints_2;
		// Mat ORB_descriptors_1, ORB_descriptors_2;
		// vector<vector<DMatch> > ORB_matches;
		// // vector<DMatch> ORB_matches;
		// vector<DMatch> ORB_good_matches;
		// Ptr<ORB> ORB_detector = ORB::create();
		// Ptr<DescriptorMatcher> ORB_matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );
		// // cout << "Creating ORB" << endl;
		// // Ptr<DescriptorMatcher> ORB_matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
		// ORB_detector -> detectAndCompute(frame1_gray, noArray(), ORB_keypoints_1, ORB_descriptors_1);
		// ORB_detector -> detectAndCompute(frame2_gray, noArray(), ORB_keypoints_2, ORB_descriptors_2);
		// // cout << "Detect and compute..." << endl;
		// // ORB_matcher -> match ( ORB_descriptors_1, ORB_descriptors_2, ORB_matches );
		// ORB_matcher->knnMatch(ORB_descriptors_1, ORB_descriptors_2, ORB_matches, 2);
		// // cout<< "matching..." << endl;
		// // only get good matching points using Lowe's ratio test
		// for (int i = 0; i < ORB_matches.size(); ++i)
		// {
		// 	if (ORB_matches[i][0].distance < ratio * ORB_matches[i][1].distance)
		// 		ORB_good_matches.push_back(ORB_matches[i][0]);
		// 	// if (ORB_matches[i].distance < ratio * ORB_matches[i].distance)
		// 	// 	ORB_good_matches.push_back(ORB_matches[i]);
		// }
		// drawMatches(frame1, ORB_keypoints_1, frame2, ORB_keypoints_2, ORB_good_matches, result, Scalar(0, 0, 255), Scalar(0, 0, 255));
		// imshow("ORB matches",result); waitKey(0);
		// // cout << "found good matches..." << endl;
		// vector<Point2f> ORB_obj, ORB_scene;
		// for (size_t i = 0; i < ORB_good_matches.size(); i++){
		// 	ORB_obj.push_back(ORB_keypoints_1[ORB_good_matches[i].queryIdx].pt);
		// 	ORB_scene.push_back(ORB_keypoints_2[ORB_good_matches[i].trainIdx].pt);
		// }
		// cout << "pushing back into obj and scene..." << endl;
		// Mat drawing_orb = frame1.clone();
		// for(size_t i=0; i<obj.size(); i++)
		// 	circle(drawing_orb,obj[i],1,colorTab[0],5,8,0);
		// for(size_t i=0; i<ORB_obj.size();i++)
		// 	circle(drawing_orb,ORB_obj[i],1,colorTab[1],5,8,0);
		// cout << "drawing points..." << endl;
		// imshow("SIFT vs ORB",drawing_orb); waitKey(0);

		/* AKAZE Feature Detector */
		// vector<KeyPoint> AKAZE_keypoints_1, AKAZE_keypoints_2;
		// Mat AKAZE_descriptors_1, AKAZE_descriptors_2;
		// vector<vector<DMatch> > AKAZE_matches;
		// // vector<DMatch> ORB_matches;
		// vector<DMatch> AKAZE_good_matches;
		// Ptr<AKAZE> AKAZE_detector = AKAZE::create();
		// // Ptr<DescriptorMatcher> AKAZE_matcher  = DescriptorMatcher::create ( "" );
		// BFMatcher AKAZE_matcher(NORM_HAMMING);
		// // cout << "Creating ORB" << endl;
		// // Ptr<DescriptorMatcher> ORB_matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
		// AKAZE_detector -> detectAndCompute(frame1_gray, noArray(), AKAZE_keypoints_1, AKAZE_descriptors_1);
		// AKAZE_detector -> detectAndCompute(frame2_gray, noArray(), AKAZE_keypoints_2, AKAZE_descriptors_2);
		// // cout << "Detect and compute..." << endl;
		// // ORB_matcher -> match ( ORB_descriptors_1, ORB_descriptors_2, ORB_matches );
		// AKAZE_matcher.knnMatch(AKAZE_descriptors_1, AKAZE_descriptors_2, AKAZE_matches, 2);
		// // cout<< "matching..." << endl;
		// // only get good matching points using Lowe's ratio test
		// for (int i = 0; i < AKAZE_matches.size(); ++i)
		// {
		// 	if (AKAZE_matches[i][0].distance < ratio * AKAZE_matches[i][1].distance)
		// 		AKAZE_good_matches.push_back(AKAZE_matches[i][0]);
		// 	// if (ORB_matches[i].distance < ratio * ORB_matches[i].distance)
		// 	// 	ORB_good_matches.push_back(ORB_matches[i]);
		// }
		// drawMatches(frame1, AKAZE_keypoints_1, frame2, AKAZE_keypoints_2, AKAZE_good_matches, result, Scalar(0, 0, 255), Scalar(0, 0, 255));
		// imshow("AKAZE matches",result); waitKey(0);
		// // cout << "found good matches..." << endl;
		// vector<Point2f> AKAZE_obj, AKAZE_scene;
		// for (size_t i = 0; i < AKAZE_good_matches.size(); i++){
		// 	AKAZE_obj.push_back(AKAZE_keypoints_1[AKAZE_good_matches[i].queryIdx].pt);
		// 	AKAZE_scene.push_back(AKAZE_keypoints_2[AKAZE_good_matches[i].trainIdx].pt);
		// }
		// // cout << "pushing back into obj and scene..." << endl;
		// // Mat drawing_akaze = frame1.clone();
		// // for(size_t i=0; i<obj.size(); i++)
		// // 	circle(drawing_akaze,obj[i],1,colorTab[0],5,8,0);
		// // for(size_t i=0; i<AKAZE_obj.size();i++)
		// // 	circle(drawing_akaze,AKAZE_obj[i],1,colorTab[1],5,8,0);
		// // cout << "drawing points..." << endl;
		// // imshow("SIFT vs AKAZE",drawing_akaze); waitKey(0);

		// Mat drawing_comparison = frame1.clone();
		// for(size_t i=0; i<obj.size(); i++)
		// 	circle(drawing_comparison,obj[i],1,colorTab[0],5,8,0);
		// for(size_t i=0; i<AKAZE_obj.size();i++)
		// 	circle(drawing_comparison,AKAZE_obj[i],1,colorTab[1],5,8,0);
		// for(size_t i=0; i<ORB_obj.size();i++)
		// 	circle(drawing_comparison,ORB_obj[i],1,colorTab[2],5,8,0);
		// imshow("SIFT vs AKAZE vs ORB",drawing_comparison); waitKey(0);
		
		return 0;
	}

	// initalizers
	float stitching_program::ratio {0.7};						// As in Lowe's paper, can be tuned (default 0.8)
	int stitching_program::left_row{20};						// pts on left edge, nuumbers of rows of pts
	int stitching_program::left_col{2};							// pts on left edge, numbers of cols of pts
	int stitching_program::middle_row{10};						// pts on middle, number of rows of pts
	int stitching_program::middle_col{8};						// pts on middle, number of cols of pts
	int stitching_program::right_row{20};						// pts on right edge, numbers of rows of pts
	int stitching_program::right_col{2};						// pts on right edge, numbers of rows of pts
	int stitching_program::width_allowance{80};					// allowance for the extra width of blended frame to n frame size
	float stitching_program::perc_width_fixed{0.25};			// % of blended frame width to be used for mapping fixed pts
	float stitching_program::perc_width_moving{0.25};			// % of n frame width to be used for mapping moving pts

	int stitching_program::Min_Num_Clusters{20};					// Min. Number of Cluster Points
	int stitching_program::Min_Num_MatchedFeaturePoints{50};	// Min. Number of Matched Feature Points
	float stitching_program::Avg_Dist_Btw_Clusters{400};		// Min. Average Distance between each Cluster Point

	float stitching_program::get_ratio(){ return stitching_program::ratio;}
	void stitching_program::change_ratio(int r) {
		stitching_program::ratio = r;
	}
	void stitching_program::get_controlpts(){
		cout<<"Left edge \t: (" << stitching_program::left_row << " x " << stitching_program::left_col << " )" << endl;
		cout<<"Middle region \t: (" << stitching_program::middle_row << " x " << stitching_program::middle_col << " )" << endl;
		cout<<"Right edge \t: (" << stitching_program::right_row << " x " << stitching_program::right_col << " )" << endl;
	}
	void stitching_program::change_controlpts(int l_r,int l_c, int m_r, int m_c, int r_r, int r_c){
		stitching_program::left_row = l_r;
		stitching_program::right_col = l_c;
		stitching_program::middle_row = m_r;
		stitching_program::middle_col = m_c;
		stitching_program::right_row = r_r;
		stitching_program::right_col = r_c;
	}
	int stitching_program::get_width_allowance(){return stitching_program::width_allowance;}
	void stitching_program::change_width_allowance(int width){
		stitching_program::width_allowance = width;
	}
	float stitching_program::get_perc_width_fixed(){return stitching_program::perc_width_fixed;}
	void stitching_program::change_perc_width_fixed(float perc){
		stitching_program::perc_width_fixed = perc;
	}
	float stitching_program::get_perc_width_moving(){return stitching_program::perc_width_moving;}
	void stitching_program::change_perc_width_moving(float perc){
		stitching_program::perc_width_moving = perc;
	}
}