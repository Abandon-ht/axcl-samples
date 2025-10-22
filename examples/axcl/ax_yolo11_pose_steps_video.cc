/*
 * AXERA is pleased to support the open source community by making ax-samples available.
 *
 * Copyright (c) 2022, AXERA Semiconductor (Shanghai) Co., Ltd. All rights reserved.
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
 * Note: For the YOLO11 series exported by the ultralytics project.
 * Author: ZHEQIUSHUI
 * Author: QQC
 * Modified to process video input with multi-threading.
 */

#include <cstdio>
#include <cstring>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include "base/common.hpp"
#include "base/detection.hpp"
#include "utilities/args.hpp"
#include "utilities/cmdline.hpp"
#include "utilities/file.hpp"
#include "utilities/timer.hpp"
#include <axcl.h>
#include "ax_model_runner/ax_model_runner_axcl.hpp"

const int DEFAULT_IMG_H = 640;
const int DEFAULT_IMG_W = 640;

const char *CLASS_NAMES[] = {
    "person",
};
const std::vector<std::vector<uint8_t>> KPS_COLORS = {
    {0, 255, 0},    {0, 255, 0},    {0, 255, 0},    {0, 255, 0},    {0, 255, 0},   {255, 128, 0},
    {255, 128, 0},  {255, 128, 0},  {255, 128, 0},  {255, 128, 0},  {255, 128, 0}, {51, 153, 255},
    {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255}};
const std::vector<std::vector<uint8_t>> LIMB_COLORS = {
    {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {255, 51, 255}, {255, 51, 255}, {255, 51, 255},
    {255, 128, 0},  {255, 128, 0},  {255, 128, 0},  {255, 128, 0},  {255, 128, 0},  {0, 255, 0},    {0, 255, 0},
    {0, 255, 0},    {0, 255, 0},    {0, 255, 0},    {0, 255, 0},    {0, 255, 0}};
const std::vector<std::vector<uint8_t>> SKELETON = {{16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13}, {6, 12}, {7, 13},
                                                    {6, 7},   {6, 8},   {7, 9},   {8, 10},  {9, 11},  {2, 3},  {1, 2},
                                                    {1, 3},   {2, 4},   {3, 5},   {4, 6},   {5, 7}};

int NUM_CLASS                = 1;
int NUM_POINT                = 17;
const int DEFAULT_LOOP_COUNT = 1;
const float PROB_THRESHOLD   = 0.45f;
const float NMS_THRESHOLD    = 0.45f;
const int QUEUE_SIZE         = 2;

// Frame queue structure
class FrameQueue {
private:
    std::queue<cv::Mat> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    const size_t max_size_;
    std::atomic<bool> stop_flag_;

public:
    FrameQueue(size_t max_size = QUEUE_SIZE) : max_size_(max_size), stop_flag_(false)
    {
    }

    void push(const cv::Mat &frame)
    {
        std::unique_lock<std::mutex> lock(mutex_);

        // If queue is full, remove oldest frame
        while (queue_.size() >= max_size_ && !stop_flag_) {
            queue_.pop();
        }

        if (!stop_flag_) {
            queue_.push(frame.clone());
            cv_.notify_one();
        }
    }

    bool pop(cv::Mat &frame)
    {
        std::unique_lock<std::mutex> lock(mutex_);

        cv_.wait(lock, [this] { return !queue_.empty() || stop_flag_; });

        if (queue_.empty()) {
            return false;  // Queue is empty and stop flag is set
        }

        frame = queue_.front();
        queue_.pop();
        return true;
    }

    void stop()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_flag_ = true;
        cv_.notify_all();
    }

    size_t size()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }
};

namespace ax {
void post_process(const ax_runner_tensor_t *output, const int nOutputSize, cv::Mat &mat, int input_w, int input_h,
                  const std::vector<float> &time_costs)
{
    std::vector<detection::Object> proposals;
    std::vector<detection::Object> objects;

    float *output_ptr[3]     = {(float *)output[0].pVirAddr,   // 1*80*80*65
                                (float *)output[1].pVirAddr,   // 1*40*40*65
                                (float *)output[2].pVirAddr};  // 1*20*20*65
    float *output_kps_ptr[3] = {(float *)output[3].pVirAddr,   // 1*80*80*51
                                (float *)output[4].pVirAddr,   // 1*40*40*51
                                (float *)output[5].pVirAddr};  // 1*20*20*51

    for (int i = 0; i < 3; ++i) {
        auto feat_ptr     = output_ptr[i];
        auto feat_kps_ptr = output_kps_ptr[i];
        int32_t stride    = (1 << i) * 8;
        detection::generate_proposals_yolov8_pose_native(stride, feat_ptr, feat_kps_ptr, PROB_THRESHOLD, proposals,
                                                         input_w, input_h, NUM_POINT, NUM_CLASS);
    }

    detection::get_out_bbox_kps(proposals, objects, NMS_THRESHOLD, input_h, input_w, mat.rows, mat.cols);
    detection::draw_keypoints(mat, objects, KPS_COLORS, LIMB_COLORS, SKELETON, "yolo11_pose_out");
}

bool run_model(const std::string &model, const std::vector<uint8_t> &data, cv::Mat &mat, int input_h, int input_w)
{
    static ax_runner_axcl runner;  // Make runner static to initialize once
    static bool is_initialized = false;

    if (!is_initialized) {
        int ret = runner.init(model.c_str());
        if (ret != 0) {
            fprintf(stderr, "init ax model runner failed.\n");
            return false;
        }
        is_initialized = true;
    }
    // 2. insert input
    memcpy(runner.get_input(0).pVirAddr, data.data(), data.size());

    // 9. run model
    std::vector<float> time_costs = {0};
    int ret                       = runner.inference();
    if (ret != 0) {
        fprintf(stderr, "Model inference failed.\n");
        return false;
    }

    // 10. get result
    post_process(runner.get_outputs_ptr(0), runner.get_num_outputs(), mat, input_w, input_h, time_costs);

    return true;
}
}  // namespace ax

// Frame capture thread function
void captureFrames(cv::VideoCapture &cap, FrameQueue &frame_queue, std::atomic<bool> &capture_stop)
{
    cv::Mat frame;

    fprintf(stdout, "Frame capture thread started.\n");

    while (!capture_stop) {
        timer t_read;
        cap >> frame;
        float read_time = t_read.cost();

        if (frame.empty()) {
            fprintf(stdout, "End of video stream or error reading frame.\n");
            capture_stop = true;
            break;
        }
        cv::flip(frame, frame, 1);
        frame_queue.push(frame);

        // Optional: Print capture stats
        if (frame_queue.size() >= QUEUE_SIZE) {
            // Queue is full, frames are being dropped
        }

        fprintf(stdout, "Frame captured in %.2f ms, queue size: %zu\n", read_time, frame_queue.size());
    }

    frame_queue.stop();
    fprintf(stdout, "Frame capture thread stopped.\n");
}

int main(int argc, char *argv[])
{
    cmdline::parser cmd;
    cmd.add<std::string>("model", 'm', "joint file(a.k.a. joint model)", true, "");
    cmd.add<std::string>("video", 'v', "video file or camera index (e.g., 0)", true, "");
    cmd.add<std::string>("size", 'g', "input_h, input_w", false,
                         std::to_string(DEFAULT_IMG_H) + "," + std::to_string(DEFAULT_IMG_W));
    cmd.parse_check(argc, argv);

    // 0. get app args
    auto model_file      = cmd.get<std::string>("model");
    auto video_source    = cmd.get<std::string>("video");
    auto model_file_flag = utilities::file_exist(model_file);

    if (!model_file_flag) {
        auto show_error = [](const std::string &kind, const std::string &value) {
            fprintf(stderr, "Input file %s(%s) is not exist, please check it.\n", kind.c_str(), value.c_str());
        };
        if (!model_file_flag) {
            show_error("model", model_file);
        }
        return -1;
    }

    auto input_size_string        = cmd.get<std::string>("size");
    std::array<int, 2> input_size = {DEFAULT_IMG_H, DEFAULT_IMG_W};
    auto input_size_flag          = utilities::parse_string(input_size_string, input_size);

    if (!input_size_flag) {
        auto show_error = [](const std::string &kind, const std::string &value) {
            fprintf(stderr, "Input %s(%s) is not allowed, please check it.\n", kind.c_str(), value.c_str());
        };
        show_error("size", input_size_string);
        return -1;
    }

    // 1. print args
    fprintf(stdout, "--------------------------------------\n");
    fprintf(stdout, "model file : %s\n", model_file.c_str());
    fprintf(stdout, "video source : %s\n", video_source.c_str());
    fprintf(stdout, "img_h, img_w : %d %d\n", input_size[0], input_size[1]);
    fprintf(stdout, "frame queue size : %d\n", QUEUE_SIZE);
    fprintf(stdout, "--------------------------------------\n");

    // 2. Open video capture
    cv::VideoCapture cap;
    try {
        // Try to interpret video_source as an integer (camera index)
        int camera_index = std::stoi(video_source);
        cap.open(camera_index, cv::CAP_V4L2);
    } catch (const std::invalid_argument &e) {
        // If not an integer, treat it as a file path
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

    // 3. init axcl
    {
        if (auto ret = axclInit(0); 0 != ret) {
            fprintf(stderr, "Init AXCL failed{0x%08x}.\n", ret);
            return -1;
        }
        axclrtDeviceList lst;
        if (const auto ret = axclrtGetDeviceList(&lst); 0 != ret || 0 == lst.num) {
            fprintf(stderr, "Get AXCL device failed{0x%08x}, find total %d device.\n", ret, lst.num);
            return -1;
        }
        if (const auto ret = axclrtSetDevice(lst.devices[0]); 0 != ret) {
            fprintf(stderr, "Set AXCL device failed{0x%08x}.\n", ret);
            return -1;
        }
        int ret = axclrtEngineInit(AXCL_VNPU_DISABLE);
        if (0 != ret) {
            fprintf(stderr, "axclrtEngineInit %d\n", ret);
            return ret;
        }
    }

    // 4. Create frame queue and start capture thread
    FrameQueue frame_queue(QUEUE_SIZE);
    std::atomic<bool> capture_stop(false);

    std::thread capture_thread(captureFrames, std::ref(cap), std::ref(frame_queue), std::ref(capture_stop));

    // 5. Process frames in main thread
    cv::Mat frame;
    std::vector<uint8_t> resized_image;
    resized_image.resize(input_size[0] * input_size[1] * 3);

    fprintf(stdout, "Starting pose detection processing. Press 'q' or 'ESC' to quit.\n");

    while (!capture_stop) {
        timer t_total;

        // Get frame from queue
        if (!frame_queue.pop(frame)) {
            // Queue is empty and capture has stopped
            break;
        }

        timer t_letterbox;
        common::get_input_data_letterbox(frame, resized_image, input_size[0], input_size[1]);
        float letterbox_time = t_letterbox.cost();

        timer t_runmodel;
        bool ok             = ax::run_model(model_file, resized_image, frame, input_size[0], input_size[1]);
        float runmodel_time = t_runmodel.cost();

        if (!ok) {
            break;
        }

        float total_time = t_total.cost();
        fprintf(stdout, "Letterbox: %.2f ms | RunModel: %.2f ms | Total: %.2f ms | Queue size: %zu\n", letterbox_time,
                runmodel_time, total_time, frame_queue.size());

        // Check for exit key (non-blocking)
        char key = (char)cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') {
            capture_stop = true;
            break;
        }
    }

    // 6. Cleanup
    capture_stop = true;
    frame_queue.stop();

    if (capture_thread.joinable()) {
        capture_thread.join();
    }

    // Release resources
    cap.release();
    cv::destroyAllWindows();

    // 7. finalize axcl
    axclFinalize();

    fprintf(stdout, "Application terminated.\n");
    return 0;
}