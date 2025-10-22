/*
 * AXERA is pleased to support the open source community by making ax-samples available.
 *
 * Copyright (c) 2024, AXERA Semiconductor Co., Ltd. All rights reserved.
 *
 * Licensed under the BSD 3-Clause License (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/BSD-3-Clause
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an "AS IS"
 * BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and limitations under the License.
 */

/*
 * Author: LittleMouse & Updated by ChatGPT
 */

#include <axcl.h>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <opencv2/opencv.hpp>

#include "ax_model_runner/ax_model_runner_axcl.hpp"
#include "base/common.hpp"
#include "base/pose.hpp"
#include "base/detection.hpp"
#include "utilities/args.hpp"
#include "utilities/cmdline.hpp"
#include "utilities/file.hpp"
#include "utilities/timer.hpp"

const int IMG_H              = 224;
const int IMG_W              = 224;
const int HAND_JOINTS        = 21;
const int DEFAULT_IMG_H      = 192;
const int DEFAULT_IMG_W      = 192;
const int DEFAULT_LOOP_COUNT = 1;

const float PROB_THRESHOLD = 0.45f;
const float NMS_THRESHOLD  = 0.45f;

const int map_size[2]        = {24, 12};
const int strides[2]         = {8, 16};
const int anchor_size[2]     = {2, 6};
const float anchor_offset[2] = {0.5f, 0.5f};

namespace ax {

void hand_post_process(const ax_runner_tensor_t *output, const int nOutputSize, const cv::Mat &mat, int input_w,
                       int input_h, const std::vector<float> &time_costs)
{
    timer timer_postprocess;
    pose::ai_hand_parts_s ai_point_result;
    auto &info_point = output[0];
    auto &info_score = output[1];
    auto *point_ptr  = (float *)info_point.pVirAddr;
    auto *score_ptr  = (float *)info_score.pVirAddr;

    pose::post_process_hand(point_ptr, score_ptr, ai_point_result, HAND_JOINTS, IMG_H, IMG_W);

    fprintf(stdout, "Handpose post process cost: %.2f ms\n", timer_postprocess.cost());
    fprintf(stdout, "--------------------------------------\n");
    auto total_time   = std::accumulate(time_costs.begin(), time_costs.end(), 0.f);
    auto min_max_time = std::minmax_element(time_costs.begin(), time_costs.end());
    fprintf(stdout, "Repeat %d times, avg %.2f ms, max %.2f ms, min %.2f ms\n", (int)time_costs.size(),
            total_time / (float)time_costs.size(), *min_max_time.second, *min_max_time.first);
    fprintf(stdout, "--------------------------------------\n");
}

bool run_hand_model(const std::string &model, const std::vector<uint8_t> &data, const int &repeat,
                    pose::ai_hand_parts_s &out_pose, int input_h, int input_w)
{
    ax_runner_axcl runner;
    if (runner.init(model.c_str()) != 0) {
        fprintf(stderr, "init handpose model runner failed.\n");
        return false;
    }
    memcpy(runner.get_input(0).pVirAddr, data.data(), data.size());

    for (int i = 0; i < 2; ++i) runner.inference();
    for (int i = 0; i < repeat; ++i) runner.inference();

    auto &info_point = runner.get_outputs_ptr(0)[0];
    auto &info_score = runner.get_outputs_ptr(0)[1];
    float *point_ptr = (float *)info_point.pVirAddr;
    float *score_ptr = (float *)info_score.pVirAddr;

    pose::post_process_hand(point_ptr, score_ptr, out_pose, HAND_JOINTS, input_h, input_w);

    runner.release();
    return true;
}

void post_process(const ax_runner_tensor_t *output, const int nOutputSize, cv::Mat &mat,  // 改成非const
                  int input_w, int input_h, const std::vector<float> &time_costs,
                  const std::string &handpose_model_file, const int &repeat)
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

    fprintf(stdout, "Palm detection post cost: %.2f ms\n", timer_postprocess.cost());
    fprintf(stdout, "--------------------------------------\n");

    cv::Mat mat_draw = mat.clone();

    for (size_t i = 0; i < objects.size(); i++) {
        cv::Mat hand_roi;
        cv::warpAffine(mat, hand_roi, objects[i].affine_trans_mat, cv::Size(IMG_W, IMG_H));

        std::vector<uint8_t> hand_image(IMG_H * IMG_W * 3, 0);
        common::get_input_data_no_letterbox(hand_roi, hand_image, IMG_H, IMG_W, true);

        pose::ai_hand_parts_s hand_parts;
        run_hand_model(handpose_model_file, hand_image, repeat, hand_parts, IMG_H, IMG_W);

        pose::draw_result_hand_on_image(mat_draw, hand_parts, HAND_JOINTS, objects[i].affine_trans_mat_inv);
    }

    detection::draw_objects_palm(mat_draw, objects, "palm_detection");
}

bool run_model(const std::string &model, const std::vector<uint8_t> &data, const int &repeat, cv::Mat &mat, int input_h,
               int input_w, const std::string &handpose_model_file)
{
    ax_runner_axcl runner;
    if (runner.init(model.c_str()) != 0) {
        fprintf(stderr, "init palm detect model runner failed.\n");
        return false;
    }

    memcpy(runner.get_input(0).pVirAddr, data.data(), data.size());
    fprintf(stdout, "Palm detection push input done.\n");

    // warmup
    for (int i = 0; i < 2; ++i) runner.inference();

    std::vector<float> time_costs(repeat, 0);
    for (int i = 0; i < repeat; ++i) {
        runner.inference();
        time_costs[i] = runner.get_inference_time();
    }

    post_process(runner.get_outputs_ptr(0), runner.get_num_outputs(), mat, input_w, input_h, time_costs,
                 handpose_model_file, repeat);

    runner.release();
    return true;
}

}  // namespace ax

int main(int argc, char *argv[])
{
    cmdline::parser cmd;
    cmd.add<std::string>("palm_model", 'm', "palm detection joint model file", true, "");
    cmd.add<std::string>("handpose_model", 'h', "handpose joint model file", true, "");
    cmd.add<std::string>("image", 'i', "input image file", true, "");
    cmd.add<std::string>("size", 'g', "input_h,input_w", false,
                         std::to_string(DEFAULT_IMG_H) + "," + std::to_string(DEFAULT_IMG_W));
    cmd.add<int>("repeat", 'r', "repeat count", false, DEFAULT_LOOP_COUNT);
    cmd.parse_check(argc, argv);

    auto palm_model_file     = cmd.get<std::string>("palm_model");
    auto handpose_model_file = cmd.get<std::string>("handpose_model");
    auto image_file          = cmd.get<std::string>("image");

    if (!utilities::file_exist(palm_model_file) || !utilities::file_exist(handpose_model_file) ||
        !utilities::file_exist(image_file)) {
        fprintf(stderr, "Some input files do not exist.\n");
        return -1;
    }

    auto input_size_string        = cmd.get<std::string>("size");
    std::array<int, 2> input_size = {DEFAULT_IMG_H, DEFAULT_IMG_W};
    if (!utilities::parse_string(input_size_string, input_size)) {
        fprintf(stderr, "Invalid input size format: %s\n", input_size_string.c_str());
        return -1;
    }

    int repeat = cmd.get<int>("repeat");

    fprintf(stdout, "--------------------------------------\n");
    fprintf(stdout, "Palm model: %s\n", palm_model_file.c_str());
    fprintf(stdout, "Handpose model: %s\n", handpose_model_file.c_str());
    fprintf(stdout, "Image file: %s\n", image_file.c_str());
    fprintf(stdout, "Input size: %d x %d\n", input_size[0], input_size[1]);
    fprintf(stdout, "Repeat: %d\n", repeat);
    fprintf(stdout, "--------------------------------------\n");

    cv::Mat mat = cv::imread(image_file);
    if (mat.empty()) {
        fprintf(stderr, "Read image failed.\n");
        return -1;
    }

    std::vector<uint8_t> image_data(input_size[0] * input_size[1] * 3, 0);
    common::get_input_data_letterbox(mat, image_data, input_size[0], input_size[1], true);

    if (auto ret = axclInit(0); ret != 0) {
        fprintf(stderr, "Init AXCL failed{0x%8x}.\n", ret);
        return -1;
    }
    axclrtDeviceList lst;
    if (auto ret = axclrtGetDeviceList(&lst); ret != 0 || lst.num == 0) {
        fprintf(stderr, "Get AXCL device failed{0x%8x}, found %d device.\n", ret, lst.num);
        return -1;
    }
    if (auto ret = axclrtSetDevice(lst.devices[0]); ret != 0) {
        fprintf(stderr, "Set AXCL device failed{0x%8x}.\n", ret);
        return -1;
    }
    if (auto ret = axclrtEngineInit(AXCL_VNPU_DISABLE); ret != 0) {
        fprintf(stderr, "axclrtEngineInit %d\n", ret);
        return ret;
    }

    ax::run_model(palm_model_file, image_data, repeat, mat, input_size[0], input_size[1], handpose_model_file);

    axclFinalize();
    return 0;
}