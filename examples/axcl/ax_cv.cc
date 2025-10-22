#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

using namespace std;
using namespace cv;

// 用chrono测时间
double benchmark(const string &name, function<void()> func, int loops = 10) {
    auto t1 = chrono::high_resolution_clock::now();
    for (int i = 0; i < loops; i++) {
        func();
    }
    auto t2 = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = t2 - t1;
    double ms = diff.count() * 1000.0 / loops;
    cout << name << " : " << ms << " ms/loop" << endl;
    return ms;
}

int main() {
    cout << "OpenCV version: " << CV_VERSION << endl;
    cout << "Threads: " << getNumThreads() << endl;

    // 创建随机矩阵（大一些以便测试差异）
    int rows = 1080;
    int cols = 1920;
    Mat A(rows, cols, CV_32FC1);
    randu(A, Scalar(0), Scalar(1));

    Mat B(rows, cols, CV_32FC1);
    randu(B, Scalar(0), Scalar(1));

    Mat C, kernel;
    kernel = getGaussianKernel(15, 3, CV_32F) * getGaussianKernel(15, 3, CV_32F).t();

    // 热身缓存（避免第一次执行包含Cache miss）
    for (int i = 0; i < 3; i++) {
        filter2D(A, C, -1, kernel);
        add(A, B, C);
        multiply(A, B, C);
    }

    // 卷积测试
    benchmark("Gaussian filter2D", [&]() {
        filter2D(A, C, -1, kernel);
    });

    // 矩阵加法测试
    benchmark("Matrix Add", [&]() {
        add(A, B, C);
    });

    // 矩阵逐元素乘法测试
    benchmark("Matrix Multiply", [&]() {
        multiply(A, B, C);
    });

    return 0;
}