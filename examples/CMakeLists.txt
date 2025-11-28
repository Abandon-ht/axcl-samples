# AXERA is pleased to support the open source community by making ax-samples available.
#
# Copyright (c) 2024, Axera Semiconductor Co., Ltd. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
#
# Author:
#


# Set axcl library
if(WIN32)
    message(STATUS "Configuring for Windows")

    if(NOT AXCL_DIR)
        set(AXCL_DIR "C:/Users/M5Stack/Downloads/AXCL/axcl/out/axcl_win_x64")
    endif()

    include_directories(${AXCL_DIR}/include)
    link_directories(${AXCL_DIR}/lib)

    set(SPDLOG_LIB_DIR "C:/Users/M5Stack/Downloads/AXCL/axcl/3rdparty/spdlog/win64")
    link_directories(${SPDLOG_LIB_DIR}/lib)

else()
    if(NOT AXCL_DIR)
        include_directories(/usr/include/axcl/)
        link_directories(/usr/lib/axcl/)
    else()
        include_directories(${AXCL_DIR}/include)
        link_directories(${AXCL_DIR}/lib)
    endif()
endif()


# Set OpenCV library
#if(NOT OpenCV_DIR)
#    set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/3rdparty/opencv-aarch64-linux/lib/cmake/opencv4)
#endif()

find_package(OpenCV REQUIRED)
include_directories(.)
include_directories(${OpenCV_INCLUDE_DIRS})


# Add example files
add_subdirectory(${CMAKE_SOURCE_DIR}/examples/axcl)
