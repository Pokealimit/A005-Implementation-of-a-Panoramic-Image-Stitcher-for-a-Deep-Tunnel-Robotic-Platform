#include "include/Stitcher.hpp"
#include "include/data_cleaning.hpp"
//#include "LaplacianBlending.h"


using namespace A005;
int main() {
	
	/* For motion detection */
	// Data_Cleaning center("/Volumes/FatBoy/NTU/FYP/Centre.avi",1);
	// // center.Data_Cleaning::change_video_name("/Users/daweikee/Documents/NTU/FYP/Progress Meeting Slides/test_video/MSP_MRT_1.MOV");
	// // center.Data_Cleaning::change_video_name("/Users/daweikee/Documents/NTU/FYP/Progress Meeting Slides/test_video/IMG_0690.MOV");	
	// center.Data_Cleaning::Dense_Optical_Flow_Detect_Motion();
	
	
	/* for writing flatten frames */
	// for(int i=0; i<1007;i++){
	// 	ostringstream name, name1;
	// 	name1 << "/Volumes/FatBoy/NTU/FYP/moving frames/frame " << i << ".bmp";
	// 	name << "/Volumes/FatBoy/NTU/FYP/flatten frames/flatten " << i << ".bmp";
	// 	Mat unflatten = imread(name1.str());
	// 	Mat flatten = center.Data_Cleaning::Flatten_Deep_Tunnel(unflatten);
	// 	imwrite(name.str(),flatten);
	// }

	
	// Mat f1, f2, result;
	stitching_program::change_controlpts(40,2,10,8,40,2);
	stitching_program::change_width_allowance(30);
	stitching_program::change_perc_width_fixed(0.41);

	/* Check if good match before stitching */
	// f1 = imread("../Unwarped_Frames/unwarped frame 7.bmp");
	// f2 = imread("../Unwarped_Frames/unwarped frame 0.bmp");
	// rotate(f1, f1, ROTATE_90_COUNTERCLOCKWISE);
	// rotate(f2, f2, ROTATE_90_COUNTERCLOCKWISE);	
	// int fit = stitching_program::Check_Points_Distribution(f1,f2);
	// string goodfit = (fit==1) ? "good" : "Not good";
	// cout << "Good fit? :\t" <<  goodfit << endl;
	
	/* Stitching with Checking for good match */
	vector<cv::String> fn;
	// create a list of image file names
	glob("/Users/daweikee/projects/opencv_project/Unwarped_Frames/*.bmp", fn, false);
	size_t count = fn.size(); //number of bmp files in images folder
	for (size_t frame_counts = 0; frame_counts < fn.size(); ){
		// for the first stitch
		for(size_t step_size = 15; frame_counts == 0;){
			ostringstream f1_name, f2_name, result_name;
			f1_name << "../Unwarped_Frames/unwarped frame " << frame_counts+step_size << ".bmp";
			f2_name << "../Unwarped_Frames/unwarped frame " << frame_counts << ".bmp";
			Mat f1 = imread(f1_name.str());
			Mat f2 = imread(f2_name.str());
			cout << "Fetching :\t" << f1_name.str() << endl;
			cout << "Fetching :\t" << f2_name.str() << endl;
			if (f1.empty() || f2.empty()) exit(-1);
			rotate(f1, f1, ROTATE_90_COUNTERCLOCKWISE);
			rotate(f2, f2, ROTATE_90_COUNTERCLOCKWISE);
			if(stitching_program::Check_Points_Distribution(f1,f2)){
				Mat result = stitching_program::stitch2frames(f1,f2);
				result = stitching_program::remove_black_portion(result);
				if(result.empty()){
					cout << "Unable to stitch.." << endl;
					step_size--;
					continue;
				}
				result_name << "./result/stitched " << frame_counts+step_size << ".bmp";
				imwrite(result_name.str(),result);
				frame_counts += step_size;
				break;
			}
			else{
				step_size--;
				if(step_size == 0) return -1;
			}
		}

		// for subsequent stitch
		for(size_t step_size = 15; step_size > 0;){
			ostringstream f1_name, f2_name, result_name;
			f1_name << "../Unwarped_Frames/unwarped frame " << frame_counts+step_size << ".bmp";
			f2_name << "./result/stitched " << frame_counts << ".bmp";
			Mat f1 = imread(f1_name.str());
			Mat f2 = imread(f2_name.str());
			cout << "Fetching :\t" << f1_name.str() << endl;
			cout << "Fetching :\t" << f2_name.str() << endl;
			if (f2.empty()) exit(-1);
			if (f1.empty()){
				step_size--;
				continue;
			}
			rotate(f1, f1, ROTATE_90_COUNTERCLOCKWISE);
			int cutoff_frame2 = 0;
			Mat f2_crop;
			if(f2.cols < (f1.cols + stitching_program::get_width_allowance()))
				f2_crop = f2.clone();
			else{
				cutoff_frame2 = f2.cols - (f1.cols + stitching_program::get_width_allowance());
				// frame2.copyTo(frame2_crop(Rect(cutoff_frame2,0,frame2.cols,frame2.rows)));
				Rect bound(0,0,f2.cols,f2.rows);
				Rect crop_frame2(cutoff_frame2,0,f2.cols,f2.rows);
				f2_crop = f2(crop_frame2 & bound);
			}
			if(stitching_program::Check_Points_Distribution(f1,f2_crop)){
				Mat result = stitching_program::stitch2frames(f1,f2_crop);
				if(result.empty()){
					cout << "Unable to stitch..." << endl;
					step_size--;
					continue;
				}
				result = stitching_program::remove_black_portion(result);
				if(cutoff_frame2>0){
					Rect bound(0,0,f2.cols,f2.rows);
					Rect crop_frame2_left(0,0,cutoff_frame2,f2.rows);
					Mat f2_left = f2(crop_frame2_left & bound);
					cout << "f2_left type:\t" << f2_left.type() << endl;
					// imshow("f2_crop_left",f2_left);waitKey(0);
					cout << "f2_crop_left size:\t" << f2_left.size() << endl;
					Mat combined;
					hconcat(f2_left,result,combined);
					// imshow("combined",combined); waitKey(0);
					result = combined.clone();
				}
				result_name << "./result/stitched " << frame_counts+step_size << ".bmp";
				imwrite(result_name.str(),result);
				frame_counts += step_size;
				break;
			}
			else{
				step_size--;
				if(step_size == 0) return -1;
			}
		}
	}




	/* for stitching */
	// for(int i = 0 ; i < 127 ; i++){
	// 	ostringstream f1_name, f2_name, resultname;
	// 	// f1_name << "../Unwarped_Frames/unwarped frame " << (i+1)*15 << ".bmp";
	// 	f1_name << "/Volumes/FatBoy/NTU/FYP/flatten frames/flatten " << (i+1)*12 << ".bmp";
	// 	if(i==0){
	// 		// f2_name << "../Unwarped_Frames/unwarped frame " << i << ".bmp";
	// 		f2_name << "/Volumes/FatBoy/NTU/FYP/flatten frames/flatten "  << i << ".bmp";
	// 		Mat f1 = imread(f1_name.str());
	// 		Mat f2 = imread(f2_name.str());
	// 		if (f1.empty() || f2.empty()) exit(-1);
	// 		rotate(f1, f1, ROTATE_90_COUNTERCLOCKWISE);
	// 		rotate(f2, f2, ROTATE_90_COUNTERCLOCKWISE);
	// 		Mat result = stitching_program::stitch2frames(f1, f2);
	// 		result = stitching_program::remove_black_portion(result);
	// 		// imshow("combined",result); waitKey(0);
	// 		resultname << "/Volumes/FatBoy/NTU/FYP/stitched map/stitched " << i <<".bmp";
	// 		imwrite(resultname.str(),result);
	// 		continue;
	// 	}
	// 	else{
	// 		f2_name << "/Volumes/FatBoy/NTU/FYP/stitched map/stitched " << i-1 << ".bmp";
	// 		Mat f1 = imread(f1_name.str());
	// 		Mat f2 = imread(f2_name.str());
	// 		if(f1.empty() || f2.empty()) exit(-1);
	// 		rotate(f1, f1, ROTATE_90_COUNTERCLOCKWISE);
	// 		int cutoff_frame2 = 0;
	// 		Mat f2_crop;
	// 		if(f2.cols < (f1.cols + stitching_program::get_width_allowance()))
	// 			f2_crop = f2.clone();
	// 		else{
	// 			cutoff_frame2 = f2.cols - (f1.cols + stitching_program::get_width_allowance());
	// 			// frame2.copyTo(frame2_crop(Rect(cutoff_frame2,0,frame2.cols,frame2.rows)));
	// 			Rect bound(0,0,f2.cols,f2.rows);
	// 			Rect crop_frame2(cutoff_frame2,0,f2.cols,f2.rows);
	// 			f2_crop = f2(crop_frame2 & bound);
	// 	}
	// 	Mat result = stitching_program::stitch2frames(f1, f2_crop);
	// 	result = stitching_program::remove_black_portion(result);
	// 	cout << "result type : \t" << result.type() << endl;
	// 	cout << "result size:\t" << result.size() << endl;
	// 	// imshow("result",result);waitKey(0);
	// 	if(cutoff_frame2>0){
	// 		Rect bound(0,0,f2.cols,f2.rows);
	// 		Rect crop_frame2_left(0,0,cutoff_frame2,f2.rows);
	// 		Mat f2_left = f2(crop_frame2_left & bound);
	// 		cout << "f2_left type:\t" << f2_left.type() << endl;
	// 		// imshow("f2_crop_left",f2_left);waitKey(0);
	// 		cout << "f2_crop_left size:\t" << f2_left.size() << endl;
	// 		Mat combined;
	// 		hconcat(f2_left,result,combined);
	// 		// imshow("combined",combined); waitKey(0);
	// 		result = combined.clone();
	// 	}
	// 	resultname << "/Volumes/FatBoy/NTU/FYP/stitched map/stitched " << i <<".bmp";
	// 	imwrite(resultname.str(),result);
		
	// 	}
	// }




	/*
	f1 = imread("../Unwarped_Frames/unwarped frame 5.bmp");
	f2 = imread("../Unwarped_Frames/unwarped frame 0.bmp");
	// f2 = imread("./result/test_stitch0.bmp");
	// cout << "frame 2 type : " << f2.type() << endl;
	if (f1.empty()) {
		cout << "cant get frame 1" << endl;
		exit(-1);
	}
	if (f2.empty()) {
		cout << "cant get frame 2" << endl;
	}	

	rotate(f1, f1, ROTATE_90_COUNTERCLOCKWISE);
	rotate(f2, f2, ROTATE_90_COUNTERCLOCKWISE);
	//imshow("frame1", f1); imshow("frame2", f2); waitKey(0);


	int cutoff_frame2 = 0;
	Mat f2_crop;
	if(f2.cols < (f1.cols + stitching_program::get_width_allowance()))
		f2_crop = f2.clone();
	else{
		cutoff_frame2 = f2.cols - (f1.cols + stitching_program::get_width_allowance());
		// frame2.copyTo(frame2_crop(Rect(cutoff_frame2,0,frame2.cols,frame2.rows)));
		Rect bound(0,0,f2.cols,f2.rows);
		Rect crop_frame2(cutoff_frame2,0,f2.cols,f2.rows);
		f2_crop = f2(crop_frame2 & bound);
	}
	// cout << "frame2_crop x:\t" << frame2_crop.cols << endl;
	// cout << "frame2_crop y:\t" << frame2_crop.rows << endl;
	// imshow("cropped frame2",frame2_crop);waitKey(0);


	result = stitching_program::stitch2frames(f1, f2_crop);
	// imshow("result", result); waitKey(0);
	// // cout << "result type : " << result.type() << endl;
	Mat stitchedmap_clean = stitching_program::remove_black_portion(result);
	// imshow("cleaned result",stitchedmap_clean); waitKey(0);
	// // imwrite("test_stitch3.bmp",stitchedmap_clean);

	if(cutoff_frame2>0){
		Rect bound(0,0,f2.cols,f2.rows);
		Rect crop_frame2_left(0,0,cutoff_frame2,f2.rows);
		Mat f2_left = f2(crop_frame2_left & bound);
		imshow("f2_crop_left",f2_left);waitKey(0);
		cout << "f2_crop_left size:\t" << f2_left.size() << endl;
		Mat combined;
		hconcat(f2_left,stitchedmap_clean,combined);
		imshow("combined",combined); waitKey(0);
	}*/

	return 0;
}