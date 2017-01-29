cs344
=====
[Introduction to Parallel Programming] (https://www.udacity.com/course/intro-to-parallel-programming--cs344)

Thanks to https://github.com/wykvictor who had a set of instructions and has solutions set up for the problem sets.  I have not tried them out yet 

## Building on Windows 10 with Visual Studio

1. Install Visual Studio:

	- [download here](https://www.visualstudio.com/vs/).
	
	- I installed Visual Studio 2015.
	
2. Install [Cuda Toolkit](https://developer.nvidia.com/cuda-toolkit):

	- [Nvidia Instructions for Windows](http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#axzz4XAtAEBZI).
	
	- [Download here](https://developer.nvidia.com/cuda-downloads).
	
	- I installed version 8.0.

3. Install [CMake](https://cmake.org/):

	- [Download here](https://cmake.org/download/).
	
	- I installed version 3.7.2.

4. Install [OpenCV](http://opencv.org/):

	- [Instructions for Installing on Windows](http://docs.opencv.org/3.2.0/d3/d52/tutorial_windows_install.html#tutorial_windows_install_prebuilt).
	
	- [Download here](https://sourceforge.net/projects/opencvlibrary/files/opencv-win/).
	
	- I installed 3.2.12, other versions should also work.
	
	- Run the EXE to extract the files. This EXE does not have an installer. Instead, you put your files where you want, and then add [environment variables for the installation location and path to DLLs](http://docs.opencv.org/3.2.0/d3/d52/tutorial_windows_install.html#tutorial_windows_install_path).
	 
	- for an installation in c:\OpenCV3.2 the following commands need to be run in a command window.
	
	```sh
	
  	setx -m OPENCV_DIR C:\OpenCV_3.2\build\x64\vc14
  	setx -m PATH=%PATH%;%OPENCV_DIR%\bin
	
	```
	note: that you may have a different path based on architecture x86 vs x64.
	
# Test nVidia compiler Version
	
Run the following in a command window to check that CUDA compiler is set properly.
	
```sh
	
	C:\ProgramData\NVIDIA Corporation\CUDA Samples\v8.0>nvcc -V
	nvcc: NVIDIA (R) Cuda compiler driver
	Copyright (c) 2005-2016 NVIDIA Corporation
	Built on Sat_Sep__3_19:05:48_CDT_2016
	Cuda compilation tools, release 8.0, V8.0.44
	
```

# Building from a command window

[Instructions](https://msdn.microsoft.com/en-ca/library/ms235639.aspx) on how to use cl in command window - nvcc should work from there too.

####This is my first attempt at formatting a markdown file. TK 20170129
