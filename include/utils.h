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

#pragma once
#include "config.h"

#include <memx/accl/utils/rental_store.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace MX::Prepost::Util {

    /** RAII return of a RentalStore slot (see MX::Utils::RentalStore). */
    template<typename T>
    struct RentalPoolGuard {
        MX::Utils::RentalStore<T>* store = nullptr;
        T* ptr = nullptr;

        RentalPoolGuard() = default;
        RentalPoolGuard(MX::Utils::RentalStore<T>* s, T* p) : store(s), ptr(p) {}

        ~RentalPoolGuard() {
            if (ptr != nullptr && store != nullptr) {
                store->return_item(ptr);
            }
        }

        RentalPoolGuard(const RentalPoolGuard&) = delete;
        RentalPoolGuard& operator=(const RentalPoolGuard&) = delete;
        RentalPoolGuard(RentalPoolGuard&&) = delete;
        RentalPoolGuard& operator=(RentalPoolGuard&&) = delete;
    };

    template<typename T>
    RentalPoolGuard<T> make_rental_pool_guard(MX::Utils::RentalStore<T>& pool,
                                             const char* checkout_failed_message) {
        T* item = pool.checkout_item();
        if (item == nullptr) {
            throw std::runtime_error(checkout_failed_message);
        }
        return RentalPoolGuard<T>(&pool, item);
    }

    /**
     * @brief ScoreManager handles confidence score thresholding and conversion.
     * Fast Sigmoid approximation: f(x) = x / (1 + |x|)
     */
    struct ScoreManager {
        float conf_thres;
        float inv_conf_thres;
        bool fast_sigmoid;

        ScoreManager(float conf_thres_, bool fast_sigmoid_ = false) :
            conf_thres(conf_thres_), fast_sigmoid(fast_sigmoid_) {
            // Must initialize inv_conf_thres AFTER fast_sigmoid is set
            inv_conf_thres = invert(conf_thres);
        }

        /**
         * @brief Logit function (Inverse Sigmoid).
         * Maps probability [0, 1] back to the raw model output space.
         */
        float invert(float p) const {
            // Clamp p to avoid log(0) or division by zero at the boundaries
            p = std::clamp(p, 1e-7f, 1.0f - 1e-7f);

            if (fast_sigmoid) {
                float x = 2.0f * p - 1.0f;         // map [0,1] -> [-1,1]
                return x / (1.0f - std::fabs(x));  // map [-1,1] -> (-inf, inf)
            } else {
                // Standard Logit: x = ln(p / (1 - p))
                return std::log(p / (1.0f - p));
            }
        }

        /**
         * @brief Sigmoid function.
         * Maps raw model output to probability [0, 1].
         */
        float convert(float x) const {
            if (fast_sigmoid) {
                // Algebraic approximation: maps (-inf, inf) to (-1, 1)
                float res = x / (1.0f + std::fabs(x));
                // Shift and scale to (0, 1)
                return (res + 1.0f) * 0.5f;
            } else {
                // Standard Logistic Sigmoid
                return 1.0f / (1.0f + std::exp(-x));
            }
        }
    };

    // Non-maximum suppression (NMS)
    // - class_agnostic=false: boxes of different classes do not suppress each other (class-aware)
    // - class_agnostic=true: boxes suppress each other regardless of class (class-agnostic)
    std::vector<int>
    nms(const std::vector<BBox>& boxes,
        float iou_thres,
        bool class_agnostic = false,
        int max_dets = 300);

    void draw_bbox(cv::Mat& image, const BBox& bbox);

    /** Reusable candidate buffers for pose postprocess (boxes + per-instance keypoints). */
    struct BBoxScratch {
        std::vector<BBox> boxes;
    };
    struct PoseScratch {
        std::vector<BBox> boxes;
        std::vector<std::vector<Keypoint>> keypoints;
    };

    /** Reusable candidate buffers for segment postprocess (boxes + mask coefficients). */
    struct SegmentScratch {
        std::vector<BBox> boxes;
        std::vector<float*> mask_coefs;
    };

    struct LetterboxParams {
        float ratio = 1.f;
        int letterbox_w = 0;
        int letterbox_h = 0;
        int pad_left = 0;
        int pad_right = 0;
        int pad_top = 0;
        int pad_bottom = 0;
    };

    LetterboxParams compute_letterbox(int ori_w, int ori_h, int model_w, int model_h);

    cv::Mat preprocess(const cv::Mat& image,
                       int letterbox_w,
                       int letterbox_h,
                       int pad_left,
                       int pad_top,
                       int pad_right,
                       int pad_bottom);

    void draw_mask(cv::Mat& image, const Mask& mask, float alpha = 0.3f);
    int get_largest_contour_idx(const std::vector<std::vector<cv::Point>>& contours);

    int get_best_label(float& best_score,
                       float* score_buf,
                       const std::vector<int>& valid_classes,
                       float score_thres);

    void get_multilabel_scores(std::vector<std::pair<int, float>>& out_labels,
                               float* score_buf,
                               const std::vector<int>& valid_classes,
                               float score_thres);
    void get_labels_by_mode(std::vector<std::pair<int, float>>& out_labels,
                            float* score_buf,
                            const std::vector<int>& valid_classes,
                            float score_thres,
                            bool use_multi_label);

    std::array<float, 4>
    dfl(float* coord_buf, int row, int col, int stride, int model_w, int model_h);

    struct Grid {
        size_t width;
        size_t height;
    };

    /** @brief Unified structure for YOLO layer parameters.
     * All ports use -1 to indicate unused ports for a given model type.
     */
    struct LayerParams {
        int coord_port = -1;
        int conf_port = -1;
        int keypt_port = -1;
        int mask_coef_port = -1;
        int mask_proto_port = -1;  // For segment models (same for all layers)
        int port_out = -1;         // For YOLOv7
        size_t width;
        size_t height;
        size_t stride;
        std::vector<Point2f> anchors;  // Only pose models populate this
    };

    std::string normalize(std::string s);

    std::size_t levenshtein(const std::string& a, const std::string& b);
    std::string colored_diff(const std::string& input, const std::string& target);

}

namespace termcolor {
    static constexpr const char* reset = "\033[0m";
    static constexpr const char* red = "\033[31m";
    static constexpr const char* green = "\033[32m";
}
