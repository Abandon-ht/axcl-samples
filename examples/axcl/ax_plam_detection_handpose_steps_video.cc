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
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
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
const float PROB_THRESHOLD   = 0.75f;
const float NMS_THRESHOLD    = 0.45f;
const int map_size[2]        = {24, 12};
const int strides[2]         = {8, 16};
const int anchor_size[2]     = {2, 6};
const float anchor_offset[2] = {0.5f, 0.5f};
const int QUEUE_SIZE         = 2;

class FrameQueue {
private:
    std::queue<cv::Mat> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    const size_t max_size_;
    std::atomic<bool> stop_flag_;
public:
    FrameQueue(size_t max_size = QUEUE_SIZE) : max_size_(max_size), stop_flag_(false) {}
    void push(const cv::Mat &frame) {
        std::unique_lock<std::mutex> lock(mutex_);
        while (queue_.size() >= max_size_ && !stop_flag_) {
            queue_.pop();
        }
        if (!stop_flag_) {
            queue_.push(frame.clone());
            cv_.notify_one();
        }
    }
    bool pop(cv::Mat &frame) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !queue_.empty() || stop_flag_; });
        if (queue_.empty()) return false;
        frame = queue_.front();
        queue_.pop();
        return true;
    }
    void stop() {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_flag_ = true;
        cv_.notify_all();
    }
    size_t size() {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }
};

namespace ax {
bool run_hand_model(const std::string &model, const std::vector<uint8_t> &data, const int &repeat,
                    pose::ai_hand_parts_s &out_pose, int input_h, int input_w)
{
    static ax_runner_axcl runner_hand;
    static bool hand_initialized = false;
    if (!hand_initialized) {
        if (runner_hand.init(model.c_str()) != 0) {
            fprintf(stderr, "init handpose model failed.\n");
            return false;
        }
        hand_initialized = true;
    }
    memcpy(runner_hand.get_input(0).pVirAddr, data.data(), data.size());
    for (int i = 0; i < repeat; ++i) runner_hand.inference();
    auto &info_point = runner_hand.get_outputs_ptr(0)[0];
    auto &info_score = runner_hand.get_outputs_ptr(0)[1];
    float *point_ptr = (float *)info_point.pVirAddr;
    float *score_ptr = (float *)info_score.pVirAddr;
    pose::post_process_hand(point_ptr, score_ptr, out_pose, HAND_JOINTS, input_h, input_w);
    return true;
}

void post_process(const ax_runner_tensor_t *output, int input_w, int input_h,
                  cv::Mat &mat, const std::string &handpose_model_file, const int &repeat)
{
    std::vector<detection::PalmObject> proposals;
    std::vector<detection::PalmObject> objects;
    auto bboxes_ptr = (float *)output[0].pVirAddr;
    auto scores_ptr = (float *)output[1].pVirAddr;
    float prob_threshold_unsigmoid = -1.0f * (float)std::log((1.0f / PROB_THRESHOLD) - 1.0f);
    detection::generate_proposals_palm(proposals, PROB_THRESHOLD, DEFAULT_IMG_W, DEFAULT_IMG_H,
                                       scores_ptr, bboxes_ptr, 2, strides, anchor_size,
                                       anchor_offset, map_size, prob_threshold_unsigmoid);
    detection::get_out_bbox_palm(proposals, objects, NMS_THRESHOLD,
                                 input_h, input_w, mat.rows, mat.cols);

    cv::Mat mat_draw = mat;  // 输出绘制用
    for (size_t i = 0; i < objects.size(); i++) {
        cv::Mat hand_roi;
        cv::warpAffine(mat, hand_roi, objects[i].affine_trans_mat, cv::Size(IMG_W, IMG_H));
        std::vector<uint8_t> hand_image(IMG_H * IMG_W * 3, 0);
        common::get_input_data_no_letterbox(hand_roi, hand_image, IMG_H, IMG_W, true);
        pose::ai_hand_parts_s hand_parts;
        run_hand_model(handpose_model_file, hand_image, repeat, hand_parts, IMG_H, IMG_W);
        pose::draw_result_hand_on_image(mat_draw, hand_parts, HAND_JOINTS, objects[i].affine_trans_mat_inv);
    }
    mat_draw = detection::draw_objects_palm(mat_draw, objects, "palm_detection");
    cv::imshow("palm_detection", mat_draw);
}

bool run_model(const std::string &palm_model, const std::vector<uint8_t> &data,
               cv::Mat &mat, int input_h, int input_w,
               const std::string &handpose_model_file, const int &repeat)
{
    static ax_runner_axcl runner_palm;
    static bool palm_initialized = false;
    if (!palm_initialized) {
        if (runner_palm.init(palm_model.c_str()) != 0) {
            fprintf(stderr, "init palm detect model failed.\n");
            return false;
        }
        palm_initialized = true;
    }
    memcpy(runner_palm.get_input(0).pVirAddr, data.data(), data.size());
    runner_palm.inference();
    post_process(runner_palm.get_outputs_ptr(0), input_w, input_h, mat, handpose_model_file, repeat);
    return true;
}
}  // namespace ax

void captureFrames(cv::VideoCapture &cap, FrameQueue &frame_queue, std::atomic<bool> &capture_stop)
{
    cv::Mat frame;
    while (!capture_stop) {
        cap >> frame;
        if (frame.empty()) {
            capture_stop = true;
            break;
        }
        cv::flip(frame, frame, 1);
        frame_queue.push(frame);
    }
    frame_queue.stop();
}

int main(int argc, char *argv[])
{
    cmdline::parser cmd;
    cmd.add<std::string>("palm_model", 'm', "palm detection joint model file", true, "");
    cmd.add<std::string>("handpose_model", 'h', "handpose joint model file", true, "");
    cmd.add<std::string>("video", 'v', "video file or camera index", true, "");
    cmd.add<std::string>("size", 'g', "input_h,input_w", false,
                         std::to_string(DEFAULT_IMG_H) + "," + std::to_string(DEFAULT_IMG_W));
    cmd.add<int>("repeat", 'r', "repeat count", false, DEFAULT_LOOP_COUNT);
    cmd.parse_check(argc, argv);

    auto palm_model_file     = cmd.get<std::string>("palm_model");
    auto handpose_model_file = cmd.get<std::string>("handpose_model");
    auto video_source        = cmd.get<std::string>("video");

    if (!utilities::file_exist(palm_model_file) || !utilities::file_exist(handpose_model_file)) {
        fprintf(stderr, "Some model files do not exist.\n");
        return -1;
    }

    std::array<int, 2> input_size = {DEFAULT_IMG_H, DEFAULT_IMG_W};
    auto size_str = cmd.get<std::string>("size");
    utilities::parse_string(size_str, input_size);
    int repeat = cmd.get<int>("repeat");

    cv::VideoCapture cap;
    try {
        int camera_index = std::stoi(video_source);
        cap.open(camera_index, cv::CAP_V4L2);
    } catch (...) {
        cap.open(video_source);
    }
    if (!cap.isOpened()) {
        fprintf(stderr, "Error opening video source: %s\n", video_source.c_str());
        return -1;
    }

    // Optional: Set capture properties if needed
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    cap.set(cv::CAP_PROP_FPS, 30);

    if (auto ret = axclInit(0); ret != 0) return -1;
    axclrtDeviceList lst;
    if (auto ret = axclrtGetDeviceList(&lst); ret != 0 || lst.num == 0) return -1;
    if (auto ret = axclrtSetDevice(lst.devices[0]); ret != 0) return -1;
    if (auto ret = axclrtEngineInit(AXCL_VNPU_DISABLE); ret != 0) return -1;

    FrameQueue frame_queue(QUEUE_SIZE);
    std::atomic<bool> capture_stop(false);
    std::thread t_cap(captureFrames, std::ref(cap), std::ref(frame_queue), std::ref(capture_stop));

    cv::Mat frame;
    std::vector<uint8_t> resized_image(input_size[0] * input_size[1] * 3);
    while (!capture_stop) {
        if (!frame_queue.pop(frame)) break;
        common::get_input_data_letterbox(frame, resized_image, input_size[0], input_size[1], true);
        ax::run_model(palm_model_file, resized_image, frame,
                      input_size[0], input_size[1], handpose_model_file, repeat);
        char key = (char)cv::waitKey(1);
        if (key == 27 || key == 'q') {
            capture_stop = true;
            break;
        }
    }

    frame_queue.stop();
    if (t_cap.joinable()) t_cap.join();
    cap.release();
    cv::destroyAllWindows();
    axclFinalize();
    return 0;
}