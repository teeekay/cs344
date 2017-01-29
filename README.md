cs344
=====
[Introduction to Parallel Programming] (https://www.udacity.com/course/intro-to-parallel-programming--cs344)
thanks to https://github.com/wykvictor

# Building on Windows 10 with Visual Studio

* Install Visual Studio:
	[download here](https://www.visualstudio.com/vs/)
	I installed Visual Studio 2015.
	
* Install [Cuda Toolkit](https://developer.nvidia.com/cuda-toolkit):
	[Nvidia Instructions for Windows](http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#axzz4XAtAEBZI)
	[Download here](https://developer.nvidia.com/cuda-downloads)
	I installed version 8.0

* Install [CMake](https://cmake.org/):
	[Download here](https://cmake.org/download/)
	I installed version 3.7.2

* Install [OpenCV](http://opencv.org/):
	[Instructions for Installing on Windows](http://docs.opencv.org/3.2.0/d3/d52/tutorial_windows_install.html#tutorial_windows_install_prebuilt)
	[Download here](https://sourceforge.net/projects/opencvlibrary/files/opencv-win/)
	I installed 3.2.12, other versions should also work.
	* Run the EXE to extract the files. This EXE does not have an installer. Instead, you put your files where you want, and then add [environment variables for the installation location and path to DLLs](http://docs.opencv.org/3.2.0/d3/d52/tutorial_windows_install.html#tutorial_windows_install_path)
	for an installation in c:\OpenCV3.2 the following commands need to be run in a command window.
  	setx -m OPENCV_DIR C:\OpenCV_3.2\build\x64\vc14
  	setx -m PATH=%PATH%;%OPENCV_DIR%\bin
  

# C:\ProgramData\NVIDIA Corporation\CUDA Samples\v8.0>nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2016 NVIDIA Corporation
Built on Sat_Sep__3_19:05:48_CDT_2016
Cuda compilation tools, release 8.0, V8.0.44


# Building from a command window
instructions on how t use cl in command window - 
nvcc will work from there too
https://msdn.microsoft.com/en-ca/library/ms235639.aspx

