cs344
=====
[Introduction to Parallel Programming] (https://www.udacity.com/course/intro-to-parallel-programming--cs344)
thanks to https://github.com/wykvictor
On Windows
# Building on Windows 10 with Visual Studio 2015

* Installed Visual Studio 2015:
	
[Nvidia reference](http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#axzz4XAtAEBZI)

* Install Cuda 8.0:
	Also refer to above link. [download](https://developer.nvidia.com/cuda-downloads)

* Install CMake:
	The latest version is OK. [download](https://cmake.org/) 

* Install OpenCV:
	I installed 3.2.12, other versions should also work. [download](http://opencv.org/)
	* Run the EXE to extract the files. This EXE does not have an installer. Instead, you put your files where you want, and then add an environment variable - see OpenCV_DIR [this](http://docs.opencv.org/2.4/doc/tutorials/introduction/windows_install/windows_install.html#windowssetpathandenviromentvariable)
  setx -m OPENCV_DIR C:\OpenCV_3.2\build\x64\vc14
  
	* Adding the environment variable named "OpenCV_DIR" (no quotes) to the "build" subfolder in the folder where you extracted.(The exact folder you need will have one very important file in it: OpenCVConfig.cmake - this tells CMake which variables to set for you.)
  
	* Add a dir of "OpenCV binary DLLs" to Windows $PATH.(like c:/OpenCV_3.2/build/x64/vc14/bin)

# C:\ProgramData\NVIDIA Corporation\CUDA Samples\v8.0>nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2016 NVIDIA Corporation
Built on Sat_Sep__3_19:05:48_CDT_2016
Cuda compilation tools, release 8.0, V8.0.44


# Building from a command window
instructions on how t use cl in command window - 
nvcc will work from there too
https://msdn.microsoft.com/en-ca/library/ms235639.aspx

