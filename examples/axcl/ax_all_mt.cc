#include <cstdio>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <axcl.h>
#include "ax_model_runner/ax_model_runner_axcl.hpp"
#include "base/common.hpp"
#include "base/detection.hpp"
#include "base/pose.hpp"
#include "utilities/cmdline.hpp"
#include "utilities/file.hpp"
#include "utilities/timer.hpp"

// ====================== 全局配置 ======================
const int DEFAULT_IMG_H = 640;
const int DEFAULT_IMG_W = 640;
const int QUEUE_SIZE    = 2;

// ====================== 帧队列类 ======================
class FrameQueue {
    std::queue<cv::Mat> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    size_t max_size_;
    std::atomic<bool> stop_flag_;

public:
    FrameQueue(size_t max_size = QUEUE_SIZE) : max_size_(max_size), stop_flag_(false)
    {
    }
    void push(const cv::Mat &frame)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        while (queue_.size() >= max_size_ && !stop_flag_) queue_.pop();
        if (!stop_flag_) {
            queue_.push(frame.clone());
            cv_.notify_one();
        }
    }
    bool pop(cv::Mat &frame)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !queue_.empty() || stop_flag_; });
        if (queue_.empty()) return false;
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
};

// ====================== 全局共享画布 ======================
cv::Mat global_canvas;
std::mutex canvas_mutex;
std::atomic<bool> stop_flag(false);

// ====================== 人脸检测 ======================
namespace task_face {
const char *CLASS_NAMES[]  = {"face"};
int NUM_CLASS              = 1;
const float PROB_THRESHOLD = 0.45f;
const float NMS_THRESHOLD  = 0.45f;
static ax_runner_axcl runner;
static bool initialized = false;
std::string model_file;
bool init()
{
    if (initialized) return true;
    if (runner.init(model_file.c_str()) != 0) {
        fprintf(stderr, "Face model init failed\n");
        return false;
    }
    initialized = true;
    return true;
}
void post_process(const ax_runner_tensor_t *output, int nOut, cv::Mat &mat, int iw, int ih)
{
    using namespace detection;
    std::vector<Object> proposals, objects;
    for (int i = 0; i < 3; ++i) {
        auto feat_ptr = (float *)output[i].pVirAddr;
        int stride    = (1 << i) * 8;
        generate_proposals_yolov8_native(stride, feat_ptr, PROB_THRESHOLD, proposals, iw, ih, NUM_CLASS);
    }
    get_out_bbox(proposals, objects, NMS_THRESHOLD, ih, iw, mat.rows, mat.cols);
    mat = draw_objects(mat, objects, CLASS_NAMES, "Face Detection");
}
bool run(cv::Mat &mat, const std::vector<uint8_t> &data, int h, int w)
{
    if (!init()) return false;
    memcpy(runner.get_input(0).pVirAddr, data.data(), data.size());
    if (runner.inference() != 0) return false;
    post_process(runner.get_outputs_ptr(0), runner.get_num_outputs(), mat, w, h);
    return true;
}
}  // namespace task_face

// ====================== 分割任务 ======================
namespace task_seg {
const char *CLASS_NAMES[] = {
    "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
    "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"};

static const std::vector<std::vector<uint8_t>> COCO_COLORS = {
    {56, 0, 255},  {226, 255, 0}, {0, 94, 255},  {0, 37, 255},  {0, 255, 94},  {255, 226, 0}, {0, 18, 255},
    {255, 151, 0}, {170, 0, 255}, {0, 255, 56},  {255, 0, 75},  {0, 75, 255},  {0, 255, 169}, {255, 0, 207},
    {75, 255, 0},  {207, 0, 255}, {37, 0, 255},  {0, 207, 255}, {94, 0, 255},  {0, 255, 113}, {255, 18, 0},
    {255, 0, 56},  {18, 0, 255},  {0, 255, 226}, {170, 255, 0}, {255, 0, 245}, {151, 255, 0}, {132, 255, 0},
    {75, 0, 255},  {151, 0, 255}, {0, 151, 255}, {132, 0, 255}, {0, 255, 245}, {255, 132, 0}, {226, 0, 255},
    {255, 37, 0},  {207, 255, 0}, {0, 255, 207}, {94, 255, 0},  {0, 226, 255}, {56, 255, 0},  {255, 94, 0},
    {255, 113, 0}, {0, 132, 255}, {255, 0, 132}, {255, 170, 0}, {255, 0, 188}, {113, 255, 0}, {245, 0, 255},
    {113, 0, 255}, {255, 188, 0}, {0, 113, 255}, {255, 0, 0},   {0, 56, 255},  {255, 0, 113}, {0, 255, 188},
    {255, 0, 94},  {255, 0, 18},  {18, 255, 0},  {0, 255, 132}, {0, 188, 255}, {0, 245, 255}, {0, 169, 255},
    {37, 255, 0},  {255, 0, 151}, {188, 0, 255}, {0, 255, 37},  {0, 255, 0},   {255, 0, 170}, {255, 0, 37},
    {255, 75, 0},  {0, 0, 255},   {255, 207, 0}, {255, 0, 226}, {255, 245, 0}, {188, 255, 0}, {0, 255, 18},
    {0, 255, 75},  {0, 255, 151}, {255, 56, 0},  {245, 255, 0}};

const float PROB_THRESHOLD = 0.45f;
const float NMS_THRESHOLD  = 0.45f;
int NUM_CLASS              = 80;
static ax_runner_axcl runner;
static bool initialized = false;
std::string model_file;
bool init()
{
    if (initialized) return true;
    if (runner.init(model_file.c_str()) != 0) {
        fprintf(stderr, "Seg model init failed\n");
        return false;
    }
    initialized = true;
    return true;
}
void post_process(const ax_runner_tensor_t *out, int, cv::Mat &mat, int iw, int ih)
{
    using namespace detection;
    std::vector<Object> proposals, objects;
    float *out_ptr[3] = {(float *)out[0].pVirAddr, (float *)out[1].pVirAddr, (float *)out[2].pVirAddr};
    float *seg_ptr[3] = {(float *)out[3].pVirAddr, (float *)out[4].pVirAddr, (float *)out[5].pVirAddr};
    for (int i = 0; i < 3; ++i) {
        int stride = (1 << i) * 8;
        generate_proposals_yolov8_seg_native(stride, out_ptr[i], seg_ptr[i], PROB_THRESHOLD, proposals, iw, ih,
                                             NUM_CLASS);
    }
    float *mask_proto_ptr = (float *)out[6].pVirAddr;
    get_out_bbox_mask(proposals, objects, mask_proto_ptr, 32, 4, NMS_THRESHOLD, ih, iw, mat.rows, mat.cols);
    mat = draw_objects_mask(mat, objects, CLASS_NAMES, COCO_COLORS, "Segmentation");
}
bool run(cv::Mat &mat, const std::vector<uint8_t> &data, int h, int w)
{
    if (!init()) return false;
    memcpy(runner.get_input(0).pVirAddr, data.data(), data.size());
    if (runner.inference() != 0) return false;
    post_process(runner.get_outputs_ptr(0), runner.get_num_outputs(), mat, w, h);
    return true;
}
}  // namespace task_seg

// ====================== 姿态任务 ======================
namespace task_pose {
const char *CLASS_NAMES[] = {"person"};

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

int NUM_CLASS              = 1;
int NUM_POINT              = 17;
const float PROB_THRESHOLD = 0.45f;
const float NMS_THRESHOLD  = 0.45f;
static ax_runner_axcl runner;
static bool initialized = false;
std::string model_file;
bool init()
{
    if (initialized) return true;
    if (runner.init(model_file.c_str()) != 0) {
        fprintf(stderr, "Pose model init failed\n");
        return false;
    }
    initialized = true;
    return true;
}
void post_process(const ax_runner_tensor_t *out, int, cv::Mat &mat, int iw, int ih)
{
    using namespace detection;
    std::vector<Object> proposals, objects;
    float *out_ptr[3] = {(float *)out[0].pVirAddr, (float *)out[1].pVirAddr, (float *)out[2].pVirAddr};
    float *kps_ptr[3] = {(float *)out[3].pVirAddr, (float *)out[4].pVirAddr, (float *)out[5].pVirAddr};
    for (int i = 0; i < 3; ++i) {
        int stride = (1 << i) * 8;
        generate_proposals_yolov8_pose_native(stride, out_ptr[i], kps_ptr[i], PROB_THRESHOLD, proposals, iw, ih,
                                              NUM_POINT, NUM_CLASS);
    }
    get_out_bbox_kps(proposals, objects, NMS_THRESHOLD, ih, iw, mat.rows, mat.cols);
    mat = draw_keypoints(mat, objects, KPS_COLORS, LIMB_COLORS, SKELETON, "Pose");
}
bool run(cv::Mat &mat, const std::vector<uint8_t> &data, int h, int w)
{
    if (!init()) return false;
    memcpy(runner.get_input(0).pVirAddr, data.data(), data.size());
    if (runner.inference() != 0) return false;
    post_process(runner.get_outputs_ptr(0), runner.get_num_outputs(), mat, w, h);
    return true;
}
}  // namespace task_pose

// ====================== 手部任务 ======================
namespace task_hand {
const int HAND_JOINTS        = 21;
const int HAND_IMG_H         = 224;
const int HAND_IMG_W         = 224;
const float PROB_THRESHOLD   = 0.65f;
const float NMS_THRESHOLD    = 0.45f;
const int map_size[2]        = {24, 12};
const int strides[2]         = {8, 16};
const int anchor_size[2]     = {2, 6};
const float anchor_offset[2] = {0.5f, 0.5f};

static ax_runner_axcl palm_runner, hand_runner;
static bool palm_init = false, hand_init = false;
std::string palm_model_file, hand_model_file;
bool init_palm()
{
    if (palm_init) return true;
    if (palm_runner.init(palm_model_file.c_str()) != 0) {
        fprintf(stderr, "Palm model init failed\n");
        return false;
    }
    palm_init = true;
    return true;
}
bool init_hand()
{
    if (hand_init) return true;
    if (hand_runner.init(hand_model_file.c_str()) != 0) {
        fprintf(stderr, "Hand model init failed\n");
        return false;
    }
    hand_init = true;
    return true;
}

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

void post_process_palm(const ax_runner_tensor_t *out, int iw, int ih, cv::Mat &mat)
{
    using namespace detection;
    std::vector<PalmObject> proposals, objects;
    auto bboxes_ptr                = (float *)out[0].pVirAddr;
    auto scores_ptr                = (float *)out[1].pVirAddr;
    float prob_threshold_unsigmoid = -1.0f * (float)std::log((1.0f / PROB_THRESHOLD) - 1.0f);
    generate_proposals_palm(proposals, PROB_THRESHOLD, 192, 192, scores_ptr, bboxes_ptr, 2, strides, anchor_size,
                            anchor_offset, map_size, prob_threshold_unsigmoid);
    get_out_bbox_palm(proposals, objects, NMS_THRESHOLD, ih, iw, mat.rows, mat.cols);

    cv::Mat mat_draw = mat;
    for (size_t i = 0; i < objects.size(); i++) {
        cv::Mat hand_roi;
        cv::warpAffine(mat, hand_roi, objects[i].affine_trans_mat, cv::Size(HAND_IMG_W, HAND_IMG_H));
        std::vector<uint8_t> hand_image(HAND_IMG_H * HAND_IMG_W * 3);
        common::get_input_data_no_letterbox(hand_roi, hand_image, HAND_IMG_H, HAND_IMG_W, true);
        pose::ai_hand_parts_s hand_parts;
        run_hand_model(hand_model_file, hand_image, 1, hand_parts, HAND_IMG_H, HAND_IMG_W);
        pose::draw_result_hand_on_image(mat_draw, hand_parts, HAND_JOINTS, objects[i].affine_trans_mat_inv);
    }
    mat = draw_objects_palm(mat_draw, objects, "Palm detection");
}
bool run(cv::Mat &mat, const std::vector<uint8_t> &data)
{
    if (!init_palm()) return false;
    if (!init_hand()) return false;
    memcpy(palm_runner.get_input(0).pVirAddr, data.data(), data.size());
    if (palm_runner.inference() != 0) return false;
    post_process_palm(palm_runner.get_outputs_ptr(0), 192, 192, mat);
    return true;
}
}  // namespace task_hand

// ====================== 各任务线程 ======================
void face_thread(FrameQueue &fq)
{
    axclInit(0);
    axclrtDeviceList lst;
    axclrtGetDeviceList(&lst);
    axclrtSetDevice(lst.devices[0]);
    axclrtEngineInit(AXCL_VNPU_DISABLE);
    cv::Mat local;
    std::vector<uint8_t> resized(DEFAULT_IMG_H * DEFAULT_IMG_W * 3);
    while (!stop_flag) {
        if (!fq.pop(local)) break;
        {
            std::lock_guard<std::mutex> lock(canvas_mutex);
            common::get_input_data_letterbox(global_canvas, resized, DEFAULT_IMG_H, DEFAULT_IMG_W);
            task_face::run(global_canvas, resized, DEFAULT_IMG_H, DEFAULT_IMG_W);
        }
    }
}
void seg_thread(FrameQueue &fq)
{
    axclInit(0);
    axclrtDeviceList lst;
    axclrtGetDeviceList(&lst);
    axclrtSetDevice(lst.devices[1]);
    axclrtEngineInit(AXCL_VNPU_DISABLE);
    cv::Mat local;
    std::vector<uint8_t> resized(DEFAULT_IMG_H * DEFAULT_IMG_W * 3);
    while (!stop_flag) {
        if (!fq.pop(local)) break;
        {
            std::lock_guard<std::mutex> lock(canvas_mutex);
            common::get_input_data_letterbox(global_canvas, resized, DEFAULT_IMG_H, DEFAULT_IMG_W);
            task_seg::run(global_canvas, resized, DEFAULT_IMG_H, DEFAULT_IMG_W);
        }
    }
}
void pose_thread(FrameQueue &fq)
{
    axclInit(0);
    axclrtDeviceList lst;
    axclrtGetDeviceList(&lst);
    axclrtSetDevice(lst.devices[1]);
    axclrtEngineInit(AXCL_VNPU_DISABLE);
    cv::Mat local;
    std::vector<uint8_t> resized(DEFAULT_IMG_H * DEFAULT_IMG_W * 3);
    while (!stop_flag) {
        if (!fq.pop(local)) break;
        {
            std::lock_guard<std::mutex> lock(canvas_mutex);
            common::get_input_data_letterbox(global_canvas, resized, DEFAULT_IMG_H, DEFAULT_IMG_W);
            task_pose::run(global_canvas, resized, DEFAULT_IMG_H, DEFAULT_IMG_W);
        }
    }
}
void hand_thread(FrameQueue &fq)
{
    axclInit(0);
    axclrtDeviceList lst;
    axclrtGetDeviceList(&lst);
    axclrtSetDevice(lst.devices[1]);
    axclrtEngineInit(AXCL_VNPU_DISABLE);
    cv::Mat local;
    std::vector<uint8_t> hand_resized(192 * 192 * 3);
    while (!stop_flag) {
        if (!fq.pop(local)) break;
        {
            std::lock_guard<std::mutex> lock(canvas_mutex);
            common::get_input_data_letterbox(global_canvas, hand_resized, 192, 192);
            task_hand::run(global_canvas, hand_resized);
        }
    }
}

// ====================== 采集线程 ======================
void captureFrames(cv::VideoCapture &cap, FrameQueue &fq)
{
    cv::Mat frame;
    while (!stop_flag) {
        cap >> frame;
        if (frame.empty()) {
            stop_flag = true;
            break;
        }
        cv::flip(frame, frame, 1);
        {
            std::lock_guard<std::mutex> lock(canvas_mutex);
            global_canvas = frame.clone();
        }
        fq.push(frame);
    }
    fq.stop();
}

// ====================== 主程序 ======================
int main(int argc, char **argv)
{
    cmdline::parser cmd;
    cmd.add<std::string>("face_model", 'f', "face model", true, "");
    cmd.add<std::string>("seg_model", 's', "seg model", true, "");
    cmd.add<std::string>("pose_model", 'p', "pose model", true, "");
    cmd.add<std::string>("palm_model", 'm', "palm model", true, "");
    cmd.add<std::string>("hand_model", 'h', "hand model", true, "");
    cmd.add<std::string>("video", 'v', "video src", true, "");
    cmd.parse_check(argc, argv);

    task_face::model_file      = cmd.get<std::string>("face_model");
    task_seg::model_file       = cmd.get<std::string>("seg_model");
    task_pose::model_file      = cmd.get<std::string>("pose_model");
    task_hand::palm_model_file = cmd.get<std::string>("palm_model");
    task_hand::hand_model_file = cmd.get<std::string>("hand_model");

    std::string video_src = cmd.get<std::string>("video");
    cv::VideoCapture cap;
    try {
        int idx = std::stoi(video_src);
        cap.open(idx, cv::CAP_V4L2);
    } catch (...) {
        cap.open(video_src);
    }
    if (!cap.isOpened()) {
        fprintf(stderr, "Video open failed.\n");
        return -1;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    cap.set(cv::CAP_PROP_FPS, 30);

    // axclInit(0);
    // axclrtDeviceList lst;
    // axclrtGetDeviceList(&lst);
    // axclrtSetDevice(lst.devices[0]);
    // axclrtEngineInit(AXCL_VNPU_DISABLE);

    FrameQueue fq(QUEUE_SIZE);

    std::thread t_cap(captureFrames, std::ref(cap), std::ref(fq));
    // std::thread t_face(face_thread, std::ref(fq));
    // std::thread t_seg(seg_thread, std::ref(fq));
    // std::thread t_pose(pose_thread, std::ref(fq));
    std::thread t_hand(hand_thread, std::ref(fq));

    while (!stop_flag) {
        {
            std::lock_guard<std::mutex> lock(canvas_mutex);
            if (!global_canvas.empty()) cv::imshow("Fusion Output", global_canvas);
        }
        char key = (char)cv::waitKey(1);
        if (key == 27 || key == 'q') {
            stop_flag = true;
            break;
        }
    }

    fq.stop();
    if (t_cap.joinable()) t_cap.join();
    // if (t_face.joinable()) t_face.join();
    // if (t_seg.joinable()) t_seg.join();
    // if (t_pose.joinable()) t_pose.join();
    if (t_hand.joinable()) t_hand.join();

    cap.release();
    cv::destroyAllWindows();
    axclFinalize();
    return 0;
}