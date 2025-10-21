/*
 * AXERA is pleased to support the open source community by making ax-samples available.
 *
 * Copyright (c) 2024, AXERA Semiconductor Co., Ltd. All rights reserved.
 *
 * Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/BSD-3-Clause
 *
 * Unless required by applicable law or agreed to in writing, software distributed
 * under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations under the License.
 */

/*
 * Author: LittleMouse
 */

#include <axcl.h>

#include <cstdio>
#include <cstring>
#include <numeric>
#include <opencv2/opencv.hpp>

#include "ax_model_runner/ax_model_runner_axcl.hpp"
#include "base/common.hpp"
#include "base/detection.hpp"
#include "utilities/args.hpp"
#include "utilities/cmdline.hpp"
#include "utilities/file.hpp"
#include "utilities/timer.hpp"

const int DEFAULT_IMG_H = 192;
const int DEFAULT_IMG_W = 192;

const int DEFAULT_LOOP_COUNT = 1;

const float PROB_THRESHOLD = 0.45f;
const float NMS_THRESHOLD  = 0.45f;

const int map_size[2]        = {24, 12};
const int strides[2]         = {8, 16};
const int anchor_size[2]     = {2, 6};
const float anchor_offset[2] = {0.5f, 0.5f};

namespace ax {
void post_process(const ax_runner_tensor_t *output, const int nOutputSize, const cv::Mat &mat, int input_w, int input_h,
                  const std::vector<float> &time_costs)
{
    std::vector<detection::PalmObject> proposals;
    std::vector<detection::PalmObject> objects;
    timer timer_postprocess;

    auto bboxes_ptr = (float *)output[0].pVirAddr;
    auto scores_ptr = (float *)output[1].pVirAddr;

    float prob_threshold_unsigmoid = -1.0f * (float)std::log((1.0f / PROB_THRESHOLD) - 1.0f);

    detection::generate_proposals_palm(proposals, PROB_THRESHOLD, DEFAULT_IMG_W, DEFAULT_IMG_H, scores_ptr, bboxes_ptr,
                                       2, strides, anchor_size, anchor_offset, map_size, prob_threshold_unsigmoid);

    detection::get_out_bbox_palm(proposals, objects, NMS_THRESHOLD, input_h, input_w, mat.rows, mat.cols);

    fprintf(stdout, "post process cost time: %.2f ms \n", timer_postprocess.cost());
    fprintf(stdout, "--------------------------------------\n");
    auto total_time   = std::accumulate(time_costs.begin(), time_costs.end(), 0.f);
    auto min_max_time = std::minmax_element(time_costs.begin(), time_costs.end());
    fprintf(stdout, "Repeat %d times, avg time %.2f ms, max_time %.2f ms, min_time %.2f ms\n", (int)time_costs.size(),
            total_time / (float)time_costs.size(), *min_max_time.second, *min_max_time.first);
    fprintf(stdout, "--------------------------------------\n");
    fprintf(stdout, "detection num: %zu\n", objects.size());

    detection::draw_objects_palm(mat, objects, "palm_detection");
}

bool run_model(const std::string &model, const std::vector<uint8_t> &data, const int &repeat, cv::Mat &mat, int input_h,
               int input_w)
{
    // 1. init runner
    ax_runner_axcl runner;
    int ret = runner.init(model.c_str());
    if (ret != 0) {
        fprintf(stderr, "init ax model runner failed.\n");
        return false;
    }

    // 2. insert input
    memcpy(runner.get_input(0).pVirAddr, data.data(), data.size());
    fprintf(stdout, "Engine push input done.\n");
    fprintf(stdout, "--------------------------------------\n");

    // 3. warmup
    for (int i = 0; i < 2; ++i) {
        runner.inference();
    }

    // 4. run model
    std::vector<float> time_costs(repeat, 0);
    for (int i = 0; i < repeat; ++i) {
        ret           = runner.inference();
        time_costs[i] = runner.get_inference_time();
    }

    // 5. get palm detection results
    post_process(runner.get_outputs_ptr(0), runner.get_num_outputs(), mat, input_w, input_h, time_costs);

    fprintf(stdout, "--------------------------------------\n");
    runner.release();
    return true;
}
}  // namespace ax

int main(int argc, char *argv[])
{
    cmdline::parser cmd;
    cmd.add<std::string>("model", 'm', "joint file(a.k.a. joint model)", true, "");
    cmd.add<std::string>("image", 'i', "image file", true, "");
    cmd.add<std::string>("size", 'g', "input_h, input_w", false,
                         std::to_string(DEFAULT_IMG_H) + "," + std::to_string(DEFAULT_IMG_W));
    cmd.add<int>("repeat", 'r', "repeat count", false, DEFAULT_LOOP_COUNT);
    cmd.parse_check(argc, argv);

    // 0. get app args, can be removed from user's app
    auto model_file = cmd.get<std::string>("model");
    auto image_file = cmd.get<std::string>("image");

    auto model_file_flag = utilities::file_exist(model_file);
    auto image_file_flag = utilities::file_exist(image_file);

    if (!model_file_flag | !image_file_flag) {
        auto show_error = [](const std::string &kind, const std::string &value) {
            fprintf(stderr, "Input file %s(%s) does not exist.\n", kind.c_str(), value.c_str());
        };
        if (!model_file_flag) show_error("model", model_file);
        if (!image_file_flag) show_error("image", image_file);
        return -1;
    }

    auto input_size_string = cmd.get<std::string>("size");

    std::array<int, 2> input_size = {DEFAULT_IMG_H, DEFAULT_IMG_W};

    auto input_size_flag = utilities::parse_string(input_size_string, input_size);

    if (!input_size_flag) {
        fprintf(stderr, "Invalid input size format: %s\n", input_size_string.c_str());
        return -1;
    }
    auto repeat = cmd.get<int>("repeat");

    // 1. print args
    fprintf(stdout, "--------------------------------------\n");
    fprintf(stdout, "model file : %s\n", model_file.c_str());
    fprintf(stdout, "image file : %s\n", image_file.c_str());
    fprintf(stdout, "img_h, img_w : %d %d\n", input_size[0], input_size[1]);
    fprintf(stdout, "--------------------------------------\n");

    // 2. read image & resize & transpose
    std::vector<uint8_t> image(input_size[0] * input_size[1] * 3, 0);
    cv::Mat mat = cv::imread(image_file);
    if (mat.empty()) {
        fprintf(stderr, "Read image failed.\n");
        return -1;
    }
    common::get_input_data_letterbox(mat, image, input_size[0], input_size[1], true);

    // 3. init axcl
    {
        if (auto ret = axclInit(0); 0 != ret) {
            fprintf(stderr, "Init AXCL failed{0x%8x}.\n", ret);
            return -1;
        }
        axclrtDeviceList lst;
        if (const auto ret = axclrtGetDeviceList(&lst); 0 != ret || lst.num == 0) {
            fprintf(stderr, "Get AXCL device failed{0x%8x}, found %d device.\n", ret, lst.num);
            return -1;
        }
        if (const auto ret = axclrtSetDevice(lst.devices[0]); 0 != ret) {
            fprintf(stderr, "Set AXCL device failed{0x%8x}.\n", ret);
            return -1;
        }
        if (const auto ret = axclrtEngineInit(AXCL_VNPU_DISABLE); 0 != ret) {
            fprintf(stderr, "axclrtEngineInit %d\n", ret);
            return ret;
        }
    }

    // 4. -  engine model
    {
        ax::run_model(model_file, image, repeat, mat, input_size[0], input_size[1]);
    }

    // 5. finalize
    axclFinalize();
    return 0;
}