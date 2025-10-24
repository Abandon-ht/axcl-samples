#pragma once

#include "base/pose.hpp"

#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <chrono>
#include <atomic>

static std::atomic<bool> gesture_control_active(false);

// ==============================
// 串口通信模块
// ==============================

FILE *serial = nullptr;  // 串口文件指针

int serial_init(const char *device)
{
    int fd = open(device, O_RDWR | O_NOCTTY | O_SYNC);
    if (fd < 0) {
        perror("open serial port");
        return -1;
    }

    struct termios tty;
    if (tcgetattr(fd, &tty) != 0) {
        perror("tcgetattr");
        close(fd);
        return -1;
    }

    // 设置输入/输出波特率
    cfsetospeed(&tty, B115200);
    cfsetispeed(&tty, B115200);

    // 配置 8N1
    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;  // 8位数据位
    tty.c_cflag &= ~PARENB;                      // 无校验
    tty.c_cflag &= ~CSTOPB;                      // 1位停止位
    tty.c_cflag |= (CLOCAL | CREAD);             // 本地模式 + 允许读
    tty.c_cflag &= ~CRTSCTS;                     // 禁用硬件流控
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);      // 禁用软件流控
    tty.c_lflag     = 0;                         // 原始输入
    tty.c_oflag     = 0;                         // 原始输出
    tty.c_cc[VMIN]  = 0;                         // 非阻塞读取
    tty.c_cc[VTIME] = 10;                        // 读取超时（1秒）

    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
        perror("tcsetattr");
        close(fd);
        return -1;
    }

    serial = fdopen(fd, "w");
    if (!serial) {
        perror("fdopen");
        close(fd);
        return -1;
    }
    return 0;
}

void serial_write(const char *msg)
{
    if (!serial) return;
    fprintf(serial, "%s", msg);
    fflush(serial);  // 立即发送
}

void serial_close()
{
    if (serial) {
        fclose(serial);
        serial = nullptr;
    }
}

// ==============================
// 手势识别模块
// ==============================

struct GestureDef {
    std::vector<int> stat;  // 0 弯曲 , 1 伸直
    std::string label;
};

static const std::vector<GestureDef> gesture_defs = {
    {{0, 1, 0, 0, 0}, "one"},  {{0, 1, 1, 0, 0}, "two"}, {{0, 1, 1, 1, 0}, "three"},    {{0, 1, 1, 1, 1}, "four"},
    {{1, 1, 1, 1, 1}, "five"}, {{0, 0, 1, 1, 1}, "ok"},  {{1, 1, 0, 0, 1}, "love you"}, {{0, 0, 0, 0, 0}, "fist"}};

// 手指关键点索引
static const int hand_landmark_point[5][5] = {
    {0, 1, 2, 3, 4},      // Thumb
    {0, 5, 6, 7, 8},      // Index
    {0, 9, 10, 11, 12},   // Middle
    {0, 13, 14, 15, 16},  // Ring
    {0, 17, 18, 19, 20}   // Pinky
};

static double vector_included_angle(const cv::Point2f &p0, const cv::Point2f &p1, const cv::Point2f &p2,
                                    const cv::Point2f &p3)
{
    double vx1 = p1.x - p0.x;
    double vy1 = p1.y - p0.y;
    double vx2 = p3.x - p2.x;
    double vy2 = p3.y - p2.y;

    double dot_product = vx1 * vx2 + vy1 * vy2;
    double len1        = std::sqrt(vx1 * vx1 + vy1 * vy1);
    double len2        = std::sqrt(vx2 * vx2 + vy2 * vy2);

    if (len1 == 0 || len2 == 0) return 0.0;

    double cos_angle = dot_product / (len1 * len2);
    cos_angle        = std::max(-1.0, std::min(1.0, cos_angle));  // 防止越界

    return std::acos(cos_angle) * 180.0 / M_PI;
}

int check_palm_objects_size(const std::vector<detection::PalmObject> &objects, float h = 0.05f, float w = 0.05f)
{
    for (const auto &obj : objects) {
        float height = obj.rect.height;
        float width  = obj.rect.width;
        if (height > h && width > w) {
            return 0;
        }
    }
    return -1;
}

int classify_gesture(const pose::ai_hand_parts_s &hand_pose)
{
    std::vector<int> finger_stat;
    finger_stat.reserve(5);

    for (int f = 0; f < 5; ++f) {
        const int *p     = hand_landmark_point[f];
        double angle_sum = 0.0;

        for (int j = 0; j < 3; ++j) {
            cv::Point2f p0(hand_pose.keypoints[p[j]].x, hand_pose.keypoints[p[j]].y);
            cv::Point2f p1(hand_pose.keypoints[p[j + 1]].x, hand_pose.keypoints[p[j + 1]].y);
            cv::Point2f p2(hand_pose.keypoints[p[j + 1]].x, hand_pose.keypoints[p[j + 1]].y);
            cv::Point2f p3(hand_pose.keypoints[p[j + 2]].x, hand_pose.keypoints[p[j + 2]].y);
            angle_sum += vector_included_angle(p0, p1, p2, p3);
        }

        finger_stat.push_back(angle_sum > 45.0 ? 0 : 1);
    }

    for (size_t i = 0; i < gesture_defs.size(); ++i) {
        if (finger_stat == gesture_defs[i].stat) {
            return static_cast<int>(i);
        }
    }
    return -1;  // 未识别
}

// ==============================
// 动作指令模块
// ==============================

struct MotionCmd {
    int mode;
    int x;
    int y;
    int rgb;
};

static const std::map<std::string, MotionCmd> BASE_CMDS = {{"reverse", {1, 600, 0, 2}},
                                                           {"forward", {1, -600, 0, 1}},
                                                           {"shake", {0, 450, 250, 4}},
                                                           {"nod", {0, 450, 450, 3}},
                                                           {"pos", {0, 450, 150, 0}}};

static const std::vector<int> SHAKE_X = {450, 250, 100, 250, 450, 650, 900, 650, 450};
static const std::vector<int> NOD_Y   = {450, 150, 450, 850, 450};

static std::chrono::steady_clock::time_point last_action_time = std::chrono::steady_clock::now();

void send_json(const MotionCmd &cmd)
{
    char buf[128];
    snprintf(buf, sizeof(buf), "{\"mode\":%d,\"x\":%d,\"y\":%d,\"rgb\":%d}\n", cmd.mode, cmd.x, cmd.y, cmd.rgb);
    serial_write(buf);
}

void send_motion(const std::string &action, int x = 0, int y = 0)
{
    auto now     = std::chrono::steady_clock::now();
    auto diff_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_action_time).count();
    if (diff_ms < 500) return;
    last_action_time = now;

    // 姿态触发的动作（除了pos）加锁检测控制
    if (action != "pos") {
        gesture_control_active.store(true);
    }

    auto it = BASE_CMDS.find(action);
    if (it == BASE_CMDS.end()) return;
    MotionCmd cmd = it->second;

    if (action == "shake") {
        std::thread([cmd]() mutable {
            for (int xv : SHAKE_X) {
                cmd.x = xv;
                send_json(cmd);
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            gesture_control_active.store(false);  // 释放锁
        }).detach();
    } else if (action == "nod") {
        std::thread([cmd]() mutable {
            for (int yv : NOD_Y) {
                cmd.y = yv;
                send_json(cmd);
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            gesture_control_active.store(false);  // 释放锁
        }).detach();
    } else if (action == "pos") {
        if (x != -1) cmd.x = x;
        if (y != -1) cmd.y = y;
        send_json(cmd);
        // pos 是检测控制，不加锁
    } else {
        send_json(cmd);
        gesture_control_active.store(false);  // 简单动作立即释放锁
    }
}

// ==============================
// 坐标处理 & 舵机控制模块
// ==============================

void process_and_send_coordinates(const std::vector<detection::Object> &objects)
{
    static int last_x = -1;
    static int last_y = -1;

    if (objects.empty()) return;
    const detection::Object &obj = objects[0];

    int center_x = obj.rect.x + obj.rect.width / 2;
    int center_y = obj.rect.y + obj.rect.height / 2;

    float scaled_x    = 150.0f + ((center_x - 0.0f) / (1000.0f)) * (600.0f - 150.0f);
    int transformed_y = 700 - center_y;

    if (last_x >= 0 && last_y >= 0) {
        if (std::fabs(scaled_x - last_x) > 10 || std::fabs(transformed_y - last_y) > 10) {
            char buf[64];
            snprintf(buf, sizeof(buf), "{\"x\":%.2f,\"y\":%d}\n", scaled_x, transformed_y);
            serial_write(buf);
        }
    }
    last_x = static_cast<int>(scaled_x);
    last_y = transformed_y;
}

void controlServo(int centerX, int centerY)
{
    static int last_x = -1;
    static int last_y = -1;

    if (gesture_control_active.load()) {
        return;
    }

    float scaled_x    = 150.0f + ((centerX - 0.0f) / 1280.0f) * (600.0f - 150.0f);
    int transformed_y = 720 - centerY;

    int scaled_x_int = static_cast<int>(scaled_x);

    if (std::abs(scaled_x_int - last_x) <= 5 && std::abs(transformed_y - last_y) <= 5) {
        return;
    }

    send_motion("pos", scaled_x_int, transformed_y);

    last_x = scaled_x_int;
    last_y = transformed_y;
}

void process_objects_for_servo(const std::vector<detection::Object> &objects, int min_h = 100, int min_w = 100)
{
    float maxArea = 0.0f;
    int centerX = -1, centerY = -1;

    for (const auto &obj : objects) {
        int box_h = static_cast<int>(obj.rect.height);
        int box_w = static_cast<int>(obj.rect.width);

        if (box_h < min_h || box_w < min_w) continue;

        float area = static_cast<float>(box_w) * box_h;
        if (area > maxArea) {
            maxArea = area;
            centerX = obj.rect.x + box_w / 2;
            centerY = obj.rect.y + box_h / 2;
        }
    }

    if (maxArea > 0) controlServo(centerX, centerY);
}
