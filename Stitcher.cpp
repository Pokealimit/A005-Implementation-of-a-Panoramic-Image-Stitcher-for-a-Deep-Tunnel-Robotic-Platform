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

	Mat stitching_program::stitch2frames(Mat frame1, Mat frame2) {
		Mat frame1_gray, frame2_gray;													// Storing frames
		vector<Point2f> edge_points, obj, scene;													// Store points
		vector <KeyPoint> keypoints1, keypoints2;													// Storing keypoints
		Mat des1, des2,result;																		// Storing Descriptors and results (some of the mat not in used - copied over from another stitching sln)
		// Mat G(3, 3, CV_64FC1);																	// For Storing Transformation Matrix of x translation
		Ptr<SIFT> detector = SIFT::create();														// Initialise Sift Detector
		Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);	// Initialise FLANN Matcher
		vector<vector<DMatch>> matches;																// Store matches
		vector<DMatch> good_matches;																// Store good matches																			// to keep track of number of stitching

		// int cutoff_frame2;
		// Mat frame2_crop;
		// if(frame2.cols < (frame1.cols + width_allowance))
		// 	frame2_crop = frame2.clone();
		// else{
		// 	cutoff_frame2 = frame2.cols - (frame1.cols + width_allowance);
		// 	// frame2.copyTo(frame2_crop(Rect(cutoff_frame2,0,frame2.cols,frame2.rows)));
		// 	Rect crop_frame2(cutoff_frame2,0,frame2.cols,frame2.rows);
		// 	frame2_crop = frame2(crop_frame2);
		// }
		// cout << "frame2_crop x:\t" << frame2_crop.cols << endl;
		// cout << "frame2_crop y:\t" << frame2_crop.rows << endl;

		// convert to grayscale for keypoints detection
		cvtColor(frame1, frame1_gray, CV_BGR2GRAY);
		cvtColor(frame2, frame2_gray, CV_BGR2GRAY);

		edge_points.push_back(Point2f(0, 0));
		edge_points.push_back(Point2f(frame1.cols,0));
		edge_points.push_back(Point2f(0, frame1.rows));
		edge_points.push_back(Point2f(frame1.cols,frame1.rows));

		// create a mask for ROI of feature detector - as there is no need to find keypoints of previous part of stitched image
		Mat mask = Mat::zeros(Size(frame2.cols, frame2.rows), CV_8UC1);
		rectangle(mask, cvPoint(frame2.cols - frame1.cols, 0), cvPoint(frame2.cols, frame2.rows), 255, -1);
		//Mat mask = Mat::zeros(Size(frame2.cols, frame2.rows), CV_8UC1);
		//rectangle(mask, cvPoint(frame2.cols - frame1.cols / 2, 0), cvPoint(frame2.cols, frame2.rows), 255, -1);	// mask out all the pixels of (frame2 - half of frame1)
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
		drawMatches(frame1, keypoints1, frame2, keypoints2, good_matches, result, Scalar(0, 0, 255), Scalar(0, 0, 255));
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
		//removing black part
		// Mat K, J, I = frame_uncut.clone();
		// cvtColor(I, K, CV_BGR2GRAY);
		// threshold(K, J, 0, 255, THRESH_BINARY);
		// vector<vector<Point>> contours;
		// vector< Vec4i > hierarchy;
		// findContours(J, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE); // Gives the outer contours
		// Mat tmp = Mat::zeros(I.size(), CV_8U);
		// int k = 0;
		// double max = -1;
		// for (size_t i = 0; i < contours.size(); i++) // Of course in this case, There is only one external contour but I write the loop for more clarification
		// {
		// 	double area = contourArea(contours[i]);
		// 	if (area > max)
		// 	{
		// 		k = i;
		// 		max = area;
		// 	}
		// }
		// drawContours(tmp, contours, k, Scalar(255, 255, 255), -1); // You can comment this line.I wrote it just for showing the procedure
		// Rect r = boundingRect(contours[k]);
		// Mat output;
		// I(r).copyTo(output);
		// //	imshow("0", I);
		// //	imshow("1", J);
		// //	imshow("2", tmp);
		// //	imshow("3", output);
		// return output;

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

	int stitching_program::Check_Points_Distribution(Mat frame1, Mat frame2){
		Mat frame1_gray, frame2_gray;																// Storing frames
		vector<Point2f> edge_points, obj, scene;													// Store points
		vector <KeyPoint> keypoints1, keypoints2;													// Storing keypoints
		Mat des1, des2,result;																		// Storing Descriptors and results
		Ptr<SIFT> detector = SIFT::create();														// Initialise Sift Detector
		Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);	// Initialise FLANN Matcher
		vector<vector<DMatch>> matches;																// Store matches
		vector<DMatch> good_matches;																// Store good matches																			// to keep track of number of stitching

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
		drawMatches(frame1, keypoints1, frame2, keypoints2, good_matches, result, Scalar(0, 0, 255), Scalar(0, 0, 255));
		// -- Get the keypoints from the good matches
		for (size_t i = 0; i < good_matches.size(); i++){
			obj.push_back(keypoints1[good_matches[i].queryIdx].pt);
			scene.push_back(keypoints2[good_matches[i].trainIdx].pt);
		}
		
		/* K Means Clustering */
		// Mat labels, centers;
		// int K=2, attempts=10, flags=KMEANS_RANDOM_CENTERS;
		// TermCriteria tc;
		// kmeans(obj,K,labels,tc,attempts,flags,centers);
		// Mat centers_points = centers.reshape(2,centers.rows);

		// Scalar colorTab[] =
    	// {
        // Scalar(0, 0, 255),
        // Scalar(0,255,0),
        // Scalar(255,100,100),
        // Scalar(255,0,255),
        // Scalar(0,255,255)
    	// };

		// Mat drawing;
		// drawing = frame1.clone();
		// vector<Point2f> contour0, contour1;	
		
		// cout << "Centers Points :\n" << centers_points << endl;

		// // for drawing cluster center points
		// for(int i=0; i < centers_points.rows; i++)
		// 	circle(drawing,centers_points.at<Point2f>(i),1,colorTab[3],5,8,0);
		
		// // cout << "labels :" << endl << labels << endl;

		// // for drawing good matching points
		// for( int i = 0; i < obj.size(); i++ )
        // {
        //     int clusterIdx = labels.at<int>(i);
        //     // circle( drawing, obj[i], 2, colorTab[clusterIdx], FILLED, LINE_AA );
		// 	circle(drawing,obj[i],1,colorTab[clusterIdx],5,8,0);

		// 	if(clusterIdx) contour1.push_back(obj[i]);
		// 	else contour0.push_back(obj[i]);
        // }

		// // for calculating area of clusters
		// double area0 = contourArea(contour0);
		// double area1 = contourArea(contour1);
		// vector<Point> approx0, approx1;
		// approxPolyDP(contour0, approx0, 5, true);
		// approxPolyDP(contour1, approx1, 5, true);
		// double area01 = contourArea(approx0);
		// double area11 = contourArea(approx1);
		// cout << "area0 =" << area0 << endl << "area01 =" << area01 << endl << "approx poly vertices" << approx0.size() << endl;
		// cout << "area1 =" << area1 << endl << "area11 =" << area11 << endl << "approx poly vertices" << approx1.size() << endl;

		// imshow("cluster points",drawing); waitKey(0);
		// return drawing;

		/* Hierachical Clustering */
		cvflann::KMeansIndexParams kmean_params(32,100,cvflann::FLANN_CENTERS_KMEANSPP);
		// cvflann::KMeansIndexParams kmean_params(32,100,cvflann::FLANN_CENTERS_RANDOM);
		Mat1f samples(obj.size(),2);

		for(int i=0;i<obj.size();i++){
			samples(i,0) = obj[i].x;
			samples(i,1) = obj[i].y;
		}
		Mat1f centers(obj.size(),2);
		// int true_number_clusters = flann::hierarchicalClustering<cv::L2<float> >(samples,centers,kmean_params);
		int true_number_clusters = flann::hierarchicalClustering<cvflann::L2<float>>(samples,centers,kmean_params);
		// int true_number_clusters = flann::hierarchicalClustering(samples,centers,kmean_params,100);
		centers = centers.rowRange(cv::Range(0,true_number_clusters));

		Mat drawing;
		drawing = frame1.clone();
		// for drawing points matching points
		for(int i=0; i < obj.size(); i++)
			circle(drawing,obj[i],1,Scalar(0,255,0),5,8,0);		
		
		cout << "Centers Points :\n" << centers << endl;
		// for drawing cluster points
		for(int i=0; i < centers.rows; i++)
			circle(drawing,centers.at<Point2f>(i),3,Scalar(0,0,255),5,8,0);
		
		
		cout << "Number of Cluster Points :\t" << centers.rows << endl;
		cout << "Number of Matched Feature Points :\t" << obj.size() << endl;
		// imshow("cluster points",drawing); waitKey(0);

		if (centers.rows < Min_Num_Clusters || obj.size() < Min_Num_MatchedFeaturePoints)
			return 0;

		/* Calculating Average Distance Between Each Cluster Points */
		float dist_btw_clusters = 0;
		for(int i=0; i<centers.rows; i++){
			float distance = 0;
			for(int j = i; j<centers.rows; j++)
				distance += norm(centers.at<Point2f>(j) - centers.at<Point2f>(j+1));
			float avg_dist = distance / (centers.rows - i);
			dist_btw_clusters += avg_dist;
		}
		float avg_dist_btw_clusters = dist_btw_clusters / centers.rows;
		cout << "Average Dist. Btw Clusters :\t" << avg_dist_btw_clusters << endl;

		if (avg_dist_btw_clusters < Avg_Dist_Btw_Clusters)
			return 0;

		return 1;

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