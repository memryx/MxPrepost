// Copyright (c) 2026 MemryX
// SPDX-License-Identifier: MIT
// 
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <memx/accl/MxAccl.h>
#include <memx/prepost/MxPrepost.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace MX::Runtime;
using namespace MX::Prepost;

static constexpr int FPS_LOG_INTERVAL = 30;  // print FPS every X frames
static constexpr size_t QUEUE_MAX = 50;      // Queue(maxsize=50)
static constexpr int DISPLAY_GET_TIMEOUT_MS = 2000;

struct Args {
    std::vector<std::string> video_paths{"/dev/video0"};
    bool no_show = false;
    std::string dfp;
    std::string task;
};

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog
              << " -d <dfp_path> -t <task> [--video_paths <p1> <p2> ...] [--no-show]\n"
              << "Examples:\n"
              << "  " << prog
              << " -d model.dfp -t yolov8-det --video_paths /dev/video0 /dev/video2\n"
              << "  " << prog << " -d model.dfp -t yolov8-det --video_paths video.mp4 --no-show\n";
}

static bool parse_args(int argc, char** argv, Args& out) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];

        if (a == "--no-show") {
            out.no_show = true;
        } else if (a == "-d" || a == "--dfp") {
            if (i + 1 >= argc)
                return false;
            out.dfp = argv[++i];
        } else if (a == "-t" || a == "--task") {
            if (i + 1 >= argc)
                return false;
            out.task = argv[++i];
        } else if (a == "--video_paths") {
            out.video_paths.clear();

            while (i + 1 < argc) {
                std::string nxt = argv[i + 1];
                if (!nxt.empty() && nxt[0] == '-')
                    break;
                out.video_paths.push_back(nxt);
                ++i;
            }
            if (out.video_paths.empty())
                return false;
        } else {
            std::cerr << "Unknown arg: " << a << "\n";
            return false;
        }
    }
    return !(out.dfp.empty() || out.task.empty());
}

static void print_result(const Result& result) {
    // ---------------------------
    // Detection (Bounding Boxes)
    // ---------------------------
    if (!result.boxes.empty()) {
        for (const auto& box : result.boxes) {
            // xyxy: [x1, y1, x2, y2]
            std::cout << "xyxy: [" << box.xyxy[0] << ", " << box.xyxy[1] << ", " << box.xyxy[2]
                      << ", " << box.xyxy[3] << "]\n";

            // xywh: [x_center, y_center, w, h]
            std::cout << "xywh: [" << box.xywh[0] << ", " << box.xywh[1] << ", " << box.xywh[2]
                      << ", " << box.xywh[3] << "]\n";

            std::cout << "conf: " << std::fixed << std::setprecision(3) << box.conf << "\n";
            std::cout << "cls_id: " << box.cls_id << "\n";
            std::cout << "cls_name: " << box.cls_name << "\n";
            std::cout << "----\n";
        }
    }

    // ---------------------------
    // Segmentation (Masks)
    // ---------------------------
    if (!result.masks.empty()) {
        for (const auto& mask : result.masks) {
            // mask.xys is polygon list of Point2f
            std::cout << "mask cls_id: " << static_cast<int>(mask.cls_id) << "\n";
            std::cout << "num polygon points: " << mask.xys.size() << "\n";

            if (!mask.xys.empty()) {
                std::cout << "first 5 points: ";
                size_t n = std::min<size_t>(5, mask.xys.size());
                for (size_t i = 0; i < n; ++i) {
                    std::cout << "(" << mask.xys[i].x << ", " << mask.xys[i].y << ")";
                    if (i + 1 < n)
                        std::cout << ", ";
                }
                std::cout << "\n";
            }
            std::cout << "----\n";
        }
    }

    // ---------------------------
    // Pose Estimation (Keypoints)
    // ---------------------------
    if (!result.keypoints.empty()) {
        for (size_t det_id = 0; det_id < result.keypoints.size(); ++det_id) {
            const auto& kps = result.keypoints[det_id];

            std::cout << "detection " << det_id << " keypoints: " << kps.size() << "\n";
            for (size_t kp_id = 0; kp_id < kps.size(); ++kp_id) {
                const auto& kp = kps[kp_id];
                std::cout << "  kp[" << kp_id << "] = (x=" << std::fixed << std::setprecision(2)
                          << kp.xy.x << ", y=" << kp.xy.y << ", conf=" << std::setprecision(3)
                          << kp.conf << ")\n";
            }
            std::cout << "----\n";
        }
    }
}

template <typename T> class BoundedQueue {
  public:
    explicit BoundedQueue(size_t maxsize) : max_(maxsize) {
    }

    bool full() const {
        std::lock_guard<std::mutex> lk(m_);
        return q_.size() >= max_;
    }

    void push(const T& v) {
        {
            std::lock_guard<std::mutex> lk(m_);
            q_.push_back(v);
        }
        cv_.notify_one();
    }

    void push(T&& v) {
        {
            std::lock_guard<std::mutex> lk(m_);
            q_.push_back(std::move(v));
        }
        cv_.notify_one();
    }

    // Pop with timeout
    bool pop_wait(T& out, int timeout_ms) {
        std::unique_lock<std::mutex> lk(m_);
        if (!cv_.wait_for(
                    lk, std::chrono::milliseconds(timeout_ms), [&] { return !q_.empty(); })) {
            return false;
        }
        out = std::move(q_.front());
        q_.pop_front();
        return true;
    }

    // If full, drop one oldest item
    void drop_one_if_full() {
        std::lock_guard<std::mutex> lk(m_);
        if (q_.size() >= max_) {
            q_.pop_front();
        }
    }

  private:
    size_t max_;
    mutable std::mutex m_;
    std::condition_variable cv_;
    std::deque<T> q_;
};

class YoloApp {
  public:
    explicit YoloApp(const Args& args) :
        args_(args), show_(!args.no_show),
        num_streams_(static_cast<int>(args.video_paths.size())) {

        // Create per-stream queues
        cap_queue_.reserve(num_streams_);
        result_queue_.reserve(num_streams_);
        for (int i = 0; i < num_streams_; ++i) {
            cap_queue_.push_back(std::make_unique<BoundedQueue<cv::Mat>>(QUEUE_MAX));
            result_queue_.push_back(
                    std::make_unique<BoundedQueue<Result>>(QUEUE_MAX));
        }

        // Open streams and store per-stream dims
        streams_.reserve(num_streams_);
        srcs_are_cams_.resize(num_streams_, true);
        ori_w_.resize(num_streams_, 0);
        ori_h_.resize(num_streams_, 0);

        frame_count_.assign(num_streams_, 0);
        start_ms_.assign(num_streams_, 0);
        fps_number_.assign(num_streams_, 0.0f);
        history_fps_.resize(num_streams_);

        for (int i = 0; i < num_streams_; ++i) {
            const std::string& p = args.video_paths[i];
            srcs_are_cams_[i] = (p.find("/dev/video") != std::string::npos);

            cv::VideoCapture cap(p, cv::CAP_V4L2);
            if (!cap.isOpened()) {
                throw std::runtime_error("Failed to open video source: " + p);
            }

            // force fourcc mjpg and 1920x1080 and 30 fps
            if (srcs_are_cams_[i]) {
                cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
                cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
                cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
                cap.set(cv::CAP_PROP_FPS, 30);
            }

            ori_w_[i] = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
            ori_h_[i] = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

            streams_.push_back(std::move(cap));
        }
    }

    void run() {

        // Set OpenCV threads to the num_streams but no more than half the
        // available threads, and at least =1
        cv::setNumThreads( max(1,min(num_streams_, cv::getNumThreads()/2)) );

        if (show_) {
            display_thread_ = std::thread(&YoloApp::display, this);
        }

        bool local = true;
        std::vector<int> device_ids{0};
        std::array<bool, 2> use_model_shape{false, false};
        SchedulerOptions options{200, 0, 16, 21, false, 11000, false, 50, 6};

        MxAccl accl{fs::path(args_.dfp), {0}, use_model_shape};

        // Connect streams first
        for (int i = 0; i < num_streams_; ++i) {
            auto in_cb = std::bind(
                    &YoloApp::in_callback, this, std::placeholders::_1, std::placeholders::_2);
            auto out_cb = std::bind(
                    &YoloApp::out_callback, this, std::placeholders::_1, std::placeholders::_2);
            accl.connect_stream(in_cb, out_cb, i /* stream id */, 0 /* model idx */);
        }

        // Shared prepost pipeline across streams
        YoloUserConfig config;
        config.conf = 0.3f;
        config.iou = 0.4f;
        config.model_id = 0;
        //config.classmap_path = "classes.txt";
        //config.valid_classes = {0};
        //config.custom_class_labels = {"person", "vehicle", "animal"};
        // config.max_dets = 300;

        //config.override_layer_mapping[8] = {"/model.22/cv2.0/cv2.0.2/Conv_output_0", "/model.22/cv3.0/cv3.0.2/Conv_output_0"};
        //config.override_layer_mapping[16] = {"/model.22/cv2.1/cv2.1.2/Conv_output_0", "/model.22/cv3.1/cv3.1.2/Conv_output_0"};
        //config.override_layer_mapping[32] = {"/model.22/cv2.2/cv2.2.2/Conv_output_0", "/model.22/cv3.2/cv3.2.2/Conv_output_0"};


        // ===========================
        // Method 1: Throwing create()
        // ===========================
        try {
            prepost_.reset(MxPrepost::create(&accl, args_.task, config));
        } catch (const std::exception& e) {
            std::cerr << e.what() << "\n";

            // Signal shutdown
            done_.store(true);

            // Clean up display thread safely
            if (show_ && display_thread_.joinable()) {
                display_thread_.join();
            }

            return;  // exit run() cleanly
        }

        // ==================================
        // Method 2: No-throw create_safe()
        // ==================================
        // std::string err;
        // if (!MxPrepost::create_safe(&accl, args_.task, config, prepost_, err)) {
        //     std::cerr << "Failed to create MxPrepost: " << err << "\n";

        //     // Clean shutdown if display thread was started
        //     done_.store(true);
        //     if (show_ && display_thread_.joinable()) {
        //         display_thread_.join();
        //     }
        //     return;  // exit run() cleanly (no abort/core dump)
        // }

        // Allocate output buffers once based on model info (shared)
        MX::Types::MxModelInfo model_info = accl.get_model_info(0);
        ofmaps_.resize(model_info.num_out_featuremaps);
        ofmap_ptrs_.resize(model_info.num_out_featuremaps);

        for (int i = 0; i < model_info.num_out_featuremaps; ++i) {
            ofmaps_[i].resize(model_info.out_featuremap_sizes[i]);
            ofmap_ptrs_[i] = ofmaps_[i].data();
        }

        accl.start();
        accl.wait();

        done_.store(true);

        if (show_ && display_thread_.joinable()) {
            display_thread_.join();
        }

        for (auto& s : streams_)
            s.release();
    }

    float get_avg_fps(int stream_id) const {
        const auto& h = history_fps_.at(stream_id);
        if (h.empty())
            return 0.0f;
        float sum = std::accumulate(h.begin(), h.end(), 0.0f);
        return sum / static_cast<float>(h.size());
    }

    int num_streams() const {
        return num_streams_;
    }

  private:
    // in_callback returns a preprocessed frame (or None)
    bool in_callback(std::vector<const MX::Types::FeatureMap*> dst, int stream_id) {
        while (true) {
            if (done_.load())
                return false;

            cv::Mat frame;
            bool got = streams_[stream_id].read(frame);
            if (!got)
                return false;

            // rule:
            // if camera + show + cap_queue full => drop frame (continue)
            if (srcs_are_cams_[stream_id] && show_ && cap_queue_[stream_id]->full()) {
                continue;
            }

            if (show_) {
                cap_queue_[stream_id]->push(frame);  // store ORIGINAL frame (BGR)
            }

            // BGR2RGB (rgb is new Mat)
            cv::Mat rgb;
            cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);

            // Preprocess
            cv::Mat pre = prepost_->preprocess(rgb);

            // Send to MXA input
            dst[0]->set_data(reinterpret_cast<float*>(pre.data));
            return true;
        }
    }

    // out_callback postprocesses per-stream using ori dims, queues result, updates fps
    bool out_callback(std::vector<const MX::Types::FeatureMap*> mxa_outputs, int stream_id) {
        // Get output data from MXA into ofmaps_
        for (int i = 0; i < static_cast<int>(mxa_outputs.size()); ++i) {
            mxa_outputs[i]->get_data(ofmaps_[i].data());
        }

        // Postprocess into a Result, using this stream's original dims
        Result result;

        prepost_->postprocess(ofmap_ptrs_, result, ori_h_[stream_id], ori_w_[stream_id]);
        // print_result(result);

        if (show_) {
            if (result_queue_[stream_id]->full()) {
                result_queue_[stream_id]->drop_one_if_full();
            }
            result_queue_[stream_id]->push(std::move(result));
        }

        update_fps(stream_id);
        return true;
    }

    void display() {
        while (!done_.load()) {
            for (int stream_id = 0; stream_id < num_streams_; ++stream_id) {
                cv::Mat frame;
                Result result;

                bool got_f = cap_queue_[stream_id]->pop_wait(frame, DISPLAY_GET_TIMEOUT_MS);
                bool got_r = result_queue_[stream_id]->pop_wait(result, DISPLAY_GET_TIMEOUT_MS);
                if (!got_f || !got_r)
                    continue;

                // Draw detections on the frame
                cv::Mat display_img = frame;  // modify in place
                prepost_->draw(display_img, result);

                // Add FPS to frame
                std::string fps_text = "FPS: " + cv::format("%.2f", fps_number_[stream_id]);
                cv::putText(display_img,
                            fps_text,
                            cv::Point(50, 50),
                            cv::FONT_HERSHEY_SIMPLEX,
                            1.0,
                            cv::Scalar(255, 0, 0),
                            2);

                std::string window_name =
                        "Stream " + std::to_string(stream_id) + " - YOLO Detection";
                cv::imshow(window_name, display_img);
            }

            // Exit if 'q' pressed
            if (cv::waitKey(1) == 'q') {
                done_.store(true);
            }
        }

        cv::destroyAllWindows();
    }

    void update_fps(int stream_id) {
        frame_count_[stream_id] += 1;

        auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(
                              std::chrono::system_clock::now())
                              .time_since_epoch()
                              .count();

        if (frame_count_[stream_id] == 1) {
            start_ms_[stream_id] = static_cast<long long>(now_ms);
            return;
        }

        if (frame_count_[stream_id] % FPS_LOG_INTERVAL == 0) {
            for (int i = 0; i < num_streams_; ++i) {
                std::cout << "Frame cnt: " << frame_count_[i] << ", Stream " << i
                          << " => FPS: " << cv::format("%.2f", fps_number_[i]) << "\n";
            }

            // overwrite of previous message block
            std::cout << "\033[" << num_streams_ << "A" << std::flush;

            for (int i = 0; i < num_streams_; ++i) {
                history_fps_[i].push_back(fps_number_[i]);
            }

            // THEN update fps_number for THIS stream only
            long long duration_ms = static_cast<long long>(now_ms) - start_ms_[stream_id];
            if (duration_ms > 0) {
                fps_number_[stream_id] = (static_cast<float>(frame_count_[stream_id]) * 1000.0f) /
                                         static_cast<float>(duration_ms);
            }
        }
    }

  private:
    Args args_;
    bool show_ = true;
    int num_streams_ = 0;

    std::vector<cv::VideoCapture> streams_;
    std::vector<bool> srcs_are_cams_;
    std::vector<int> ori_w_;
    std::vector<int> ori_h_;

    std::vector<std::unique_ptr<BoundedQueue<cv::Mat>>> cap_queue_;
    std::vector<std::unique_ptr<BoundedQueue<Result>>> result_queue_;

    std::unique_ptr<MxPrepost> prepost_;

    // shared output buffers
    std::vector<std::vector<float>> ofmaps_;
    std::vector<float*> ofmap_ptrs_;

    std::atomic_bool done_{false};
    std::thread display_thread_;

    // FPS per stream
    std::vector<int> frame_count_;
    std::vector<long long> start_ms_;
    std::vector<float> fps_number_;
    std::vector<std::vector<float>> history_fps_;
};

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        print_usage(argv[0]);
        return 1;
    }

    try {
        YoloApp app(args);
        app.run();

        for (int i = 0; i < app.num_streams(); ++i) {
            std::cout << "\n\nFinal Avg FPS for Stream " << i << ": "
                      << cv::format("%.2f", app.get_avg_fps(i)) << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
