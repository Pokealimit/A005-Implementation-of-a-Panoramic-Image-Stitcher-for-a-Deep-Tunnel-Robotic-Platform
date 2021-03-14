# A005-Integrated-User-Interface-for-the-Control-of-a-Deep-Tunnel-Robotic-Platform

This is the repository for my Final Year Project in Nanyang Technological University (NTU) <br />
Title: A005: Integrated User Interface for the Control of a Deep Tunnel Robotic Platform

The C++ require the installation of the below module:
- OpenCV4 ( [Open Source ComputerVision Library](https://github.com/opencv/opencv) )
- Eigen ( [Matrix Class](https://eigen.tuxfamily.org/dox/group__TutorialMatrixClass.html) )
* Do include the libraries in ur own IDE/txt editor when using the repository

Below are the functions contained inside the various C++ Files:
- Source.cpp (main entry point)
- ImageStitchingOpenCV.cpp (test stitching code)
- data_cleaning.cpp (class file for detecting motion and extract moving frames & Distort to flatten Deep Tunnel Image)
- Stitcher.cpp (class file for stitching 2 frames tgt by matching,blending and warping in order to provide a straight mosaic)
