#include "include/data_cleaning.hpp"

using namespace std;
using namespace cv;
namespace A005 {

	void Data_Cleaning::Dense_Optical_Flow_Detect_Motion() {
		cout << "Video name is : " << video_name << endl;
		cout << "Orientation is : " << orient << endl;

        VideoCapture capture(video_name);
        if (!capture.isOpened()) {
            cout << "Unable to open video file 0: " << endl;
            exit(1);
        }
        cout << "Start grabbing" << endl;
        
        int count = 0;
        Mat frame1, prvs;
        capture >> frame1;
        cvtColor(frame1, prvs, COLOR_BGR2GRAY);
        // imshow("actual video",frame1);
        while(true){
            Mat frame2, next;
            capture >> frame2;
            if (frame2.empty())
                break;
            cvtColor(frame2, next, COLOR_BGR2GRAY);
            Mat flow(prvs.size(), CV_32FC2);
            // calcOpticalFlowFarneback(prvs, next, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
            calcOpticalFlowFarneback(prvs, next, flow, 0.5, 3, 25, 3, 5, 1.2, 0);
            // visualization
            Mat flow_parts[2];
            split(flow, flow_parts);
            Mat magnitude, angle, magn_norm;
            cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
            normalize(magnitude, magn_norm, 0.0f, 1.0f, NORM_MINMAX);
            cout << " magnitude matrix size :\t" << magnitude.size() << endl;
            // creating threshold mask
            Mat thresh, magnitude8;
            magnitude.convertTo(magnitude8,CV_8UC1,255.0);
            imshow("magnitude8",magnitude8);
            threshold(magnitude8, thresh, 210, 255, THRESH_BINARY);  // remove noise by setting noises below certain value to 255 (black)
            imshow("thresh",thresh);

            // If all black -> no motion
			if (countNonZero(thresh) == 0) 
				cout << "there is no motion" << endl;
			else {
				cout << "there is a motion" << endl;
                // ostringstream name;
                // name << "/Volumes/FatBoy/NTU/FYP/moving frames/frame " << count++ << ".bmp";
                // imwrite(name.str(),frame2);
            }
            angle *= ((1.f / 360.f) * (180.f / 255.f));
            //build hsv image
            Mat _hsv[3], hsv, hsv8, bgr;
            _hsv[0] = angle;
            _hsv[1] = Mat::ones(angle.size(), CV_32F);
            _hsv[2] = magn_norm;
            merge(_hsv, 3, hsv);
            hsv.convertTo(hsv8, CV_8U, 255.0);
            cvtColor(hsv8, bgr, COLOR_HSV2BGR);
            // imshow("dense optical flow video", bgr);
            // imshow("actual video",frame2);

            /* to show the dense optical flow with video */
            Mat blend;
            addWeighted(frame2,0.3,bgr,0.7,0.0,blend);
            imshow("dense flow",blend);
            int keyboard = waitKey(1);
            if (keyboard == 'q' || keyboard == 27)
                break;
            prvs = next;
        }


		return;
	}




    string Data_Cleaning::get_video_name(){return video_name;}
    int Data_Cleaning::change_video_name(string name){
        video_name = name;
        return 0;
    }
    string Data_Cleaning::get_orientation(){
        string output;
        switch (orient)
        {
        case 0:
            output = "Left";
            break;
        case 1:
            output = "Centre";
            break;
        case 2:
            output = "Right";
            break;
        default:
            output = "Invalid";
            break;
        }
        return output;
    }
    int Data_Cleaning::change_orientation(int orientation){
        if(orientation == 0 || orientation == 1 || orientation || 2)
            orient = orientation;
        else
            orient = -1;
        return 0;
    }

    Mat Data_Cleaning::Flatten_Deep_Tunnel(Mat frame){
        Mat dst;    //destination mat
        
        // Set up transformation matrix
        // For Centre Orientation
        if(orient == 1){
            Mat_<float> map_x(frame.size()), map_y(frame.size());
            for (int i = 0; i < map_x.rows; i++)
            {
                for (int j = 0; j < map_x.cols; j++)
                {
                    //Poly 4
                    map_x(i, j) = 154.0922468143963 + 0.7719802089213 * j - 0.4444148023074 * i + 0.0005102807372 * j * i + 0.0000482389365 * j * j + 0.0002746555961 * i * i - 0.0000000733884 * i * j * j - 0.0000003485579 * i * i * j - 0.0000000018574 * j * j * j + 0.0000001413314 * i * i * i + 0.0000000000165 * j * j * j * i + 0.0000000000180 * i * i * j * j - 0.0000000000534 * i * i * i * j - 0.0000000000033 * j * j * j * j - 0.0000000000418 * i * i * i * i;
                    map_y(i, j) = -107.1424138001671 + 0.4099209793614 * j + 0.8622179176736 * i - 0.0005529197790 * j * i - 0.0002509806111 * j * j + 0.0007452869698 * i * i + 0.0000003139380 * i * j * j - 0.0000001194946 * i * i * j + 0.0000000356792 * j * j * j - 0.0000004432920 * i * i * i - 0.0000000000148 * j * j * j * i + 0.0000000000473 * i * i * j * j + 0.0000000000120 * i * i * i * j - 0.0000000000063 * j * j * j * j + 0.0000000000250 * i * i * i * i;
                }
            }
            // create destination mat & unwarp frame
            remap(frame, dst, map_x, map_y, INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
            if (dst.empty()) {
                cout << "can't remap" << endl;
                exit(-1);
            }
        }
        return dst;
    }
}