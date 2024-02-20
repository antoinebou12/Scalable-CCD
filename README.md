# Scalable CCD

[![Build](https://github.com/continuous-collision-detection/scalable-ccd/actions/workflows/continuous.yml/badge.svg)](https://github.com/continuous-collision-detection/scalable-ccd/actions/workflows/continuous.yml)
[![License](https://img.shields.io/github/license/continuous-collision-detection/scalable-ccd.svg?color=blue)](https://github.com/continuous-collision-detection/scalable-ccd/blob/main/LICENSE)

Sweep and Tiniest Queue & Tight-Inclusion GPU CCD

## Getting Started

### Prerequisites

* A C/C++ compiler (at least support for C++17)
* CMake (version 3.18 or newer)
* Optionally: A CUDA-compatible GPU and the CUDA toolkit installed

### Building

The easiest way to add this project to an existing CMake project is to download it through CMake. Here is an example of how to add this project to your CMake project using [CPM](https://github.com/cpm-cmake/CPM.cmake):

```cmake
# Scalable CCD (https://github.com/continuous-collision-detection/scalable-ccd)
# License: Apache 2.0
if(TARGET scalable_ccd::scalable_ccd)
    return()
endif()

message(STATUS "Third-party: creating target 'scalable_ccd::scalable_ccd'")

set(SCALABLE_CCD_WITH_CUDA ${MY_PROJECT_WITH_CUDA} CACHE BOOL "Enable CUDA CCD" FORCE)

include(CPM)
CPMAddPackage("gh:continuous-collision-detection/scalable-ccd#${SCALABLE_CCD_GIT_TAG}")
```

where `MY_PROJECT_WITH_CUDA` is an example variable set in your project and  `SCALABLE_CCD_GIT_TAG` is set to the version of this project you want to use. This will download and add this project to CMake. You can then be linked against it using

```cmake
# Link against the Scalable CCD
target_link_libraries(my_target PRIVATE scalable_ccd::scalable_ccd)
```

where `my_target` is the name of your library/binary.

#### Dependencies

**All required dependencies are downloaded through CMake** depending on the build options.

The following libraries are used in this project:

* [Eigen](https://eigen.tuxfamily.org/): linear algebra
* [oneTBB](https://github.com/oneapi-src/oneTBB): CPU multi-threading
* [spdlog](https://github.com/gabime/spdlog): logging

##### Optional

* [CUDA](https://developer.nvidia.com/cuda-toolkit): GPU acceleration
	* Required when using the CMake option `SCALABLE_CCD_WITH_CUDA`
* [nlohmann/json](https://github.com/nlohmann/json): saving profiler data
    * Required when using the CMake option `SCALABLE_CCD_WITH_PROFILER`

## Usage

:hammer_and_wrench: **ToDo**: Write usage instructions.

## Citation

If you use this code in your project, please consider citing our paper:

```bibtex
@misc{belgrod2023time,
	title        = {Time of Impact Dataset for Continuous Collision Detection and a Scalable Conservative Algorithm},
	author       = {David Belgrod and Bolun Wang and Zachary Ferguson and Xin Zhao and Marco Attene and Daniele Panozzo and Teseo Schneider},
	year         = 2023,
	eprint       = {2112.06300},
	archiveprefix = {arXiv},
	primaryclass = {cs.GR}
}
```

## License

This project is licensed under the Apache-2.0 license - see the [LICENSE](https://github.com/continuous-collision-detection/scalable-ccd/blob/main/LICENSE) file for details.