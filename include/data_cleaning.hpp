#ifndef __data_cleaning__
#define __data_cleaning__

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <opencv2/calib3d.hpp>
#include <stdlib.h>
#include <stdio.h>

using namespace std;
using namespace cv;
namespace A005 {
	
	class  Data_Cleaning
	{	
	public:
		// Constructors
        // input path/name of video file // Orientation: 0 - left | 1 - centre | 2 - right
		Data_Cleaning(string name_of_video, int orientation) { 
            video_name = name_of_video; 
            if(orientation == 0 || orientation == 1 || orientation || 2)
                orient = orientation;
            else
                orient = -1; 
        }
		~Data_Cleaning() {}

		// See current Name of input video
		string get_video_name();
        // change input video name
        int change_video_name(string name);
        // See current video orientation input
		string get_orientation();
        // change video input orientation // Orientation: 0 - left | 1 - centre | 2 - right
        int change_orientation(int orientation);
		// detect motion using Lukas Kanade Method
		void Dense_Optical_Flow_Detect_Motion();
		// Distort input deep tunnel image to flatten before stitching
		Mat Flatten_Deep_Tunnel(Mat frame);

	private:
		string video_name;
        int orient;
	};

}

#endif