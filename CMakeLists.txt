############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################

cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(cs344)

find_package(OpenCV 3.2 REQUIRED)
find_package(CUDA 8.0 REQUIRED)

link_libraries(${OpenCV_LIBS} )

set (EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/bin/")

if(CUDA_FOUND)  
  #set build for Pascal GTX 10xx cards 
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
    -gencode arch=compute_61,code=sm_61)
  # compared to class settings, we let NVidia's FindCUDA CMake detect 
  # whether to build x64.  We tell it to support most devices, though, 
  # to make sure more people can easily run class code without knowing 
  # about this compiler argument
  #  set(CUDA_NVCC_FLAGS "
  #  -ccbin /usr/bin/clang; 
  #  -gencode;arch=compute_30,code=sm_30;  
  #  -gencode;arch=compute_35,code=sm_35;
  #  -gencode;arch=compute_35,code=compute_35;
  #  -gencode;arch=compute_20,code=sm_20")
   
  
  # add -Wextra compiler flag for gcc compilations
  if (UNIX)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler -Wextra")
    set(CMAKE_CXX_FLAGS "-stdlib=libstdc++")
  endif (UNIX)

  # add debugging to CUDA NVCC flags.  For NVidia's NSight tools.
  set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")

  # need to set fused multiply and add to off to get same results as reference calculations. see
  # https://developer.nvidia.com/sites/default/files/akamai/cuda/files/NVIDIA-CUDA-Floating-Point.pdf 
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-fmad=false)

  add_subdirectory ("Problem Sets/Problem Set 1")
  add_subdirectory ("Problem Sets/Problem Set 2")
  add_subdirectory ("Problem Sets/Problem Set 3")
  add_subdirectory ("Problem Sets/Problem Set 4")
  add_subdirectory ("Problem Sets/Problem Set 5")
  add_subdirectory ("Problem Sets/Problem Set 6")
    
else(CUDA_FOUND)
  message("CUDA is not installed on this system.")
endif()
