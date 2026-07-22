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

#include "utils.h"

#include <memx/accl/utils/macros.h>

#include <limits>
#include <stdexcept>
#include <vector>

#define FONT (cv::FONT_ITALIC)

namespace {  // anonymous namespace for internal linkage
    constexpr int DEFAULT_FONT = cv::FONT_ITALIC;

    const std::vector<cv::Scalar> TEXT_COLORS = {
            {0, 0, 0},
            {255, 255, 255},
            {255, 255, 255},
            {255, 255, 255},
            {255, 215, 0},
    };

    const std::vector<cv::Scalar> BOX_COLORS = {
            {255, 255, 0, 0.6},
            {26, 35, 126, 0.6},
            {255, 50, 50, 0.6},
            {0, 0, 0, 0.6},
            {51, 51, 51, 0.6},
    };

    // Helper to get color safely with modulo
    cv::Scalar get_color(const std::vector<cv::Scalar>& palette, int id) {
        return palette[id % palette.size()];
    }
}

namespace MX::Prepost::Util {

    std::vector<int>
    nms(const std::vector<BBox>& boxes, float iou_thres, bool class_agnostic, int max_dets) {
        if (boxes.empty())
            return {};

        const int n = boxes.size();
        const int max_det = std::max(1, max_dets);
        const int num_candidates = n;

        // 1. Sort indices based on conf scores
        std::vector<int> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](int i, int j) {
            return boxes[i].conf > boxes[j].conf;
        });

        // 2. Pre-calculate areas
        std::vector<float> areas(n);
        for (int idx : indices) {
            areas[idx] =
                    (boxes[idx].x_max - boxes[idx].x_min) * (boxes[idx].y_max - boxes[idx].y_min);
        }

        // 3. Bitset-style suppression for efficiency
        std::vector<int> suppressed(n, 0);
        std::vector<int> keep;
        keep.reserve(std::min(n, max_det));  // Pre-allocate memory

        for (int i = 0; i < num_candidates; ++i) {
            int idx_i = indices[i];
            if (suppressed[idx_i])
                continue;

            keep.push_back(idx_i);
            if ((int)keep.size() >= max_det)
                break;

            for (int j = i + 1; j < num_candidates; ++j) {
                int idx_j = indices[j];
                if (suppressed[idx_j])
                    continue;

                // If class-aware NMS, only suppress boxes of the same class
                if (!class_agnostic && boxes[idx_i].cls_id != boxes[idx_j].cls_id) {
                    continue;  // Skip if different classes
                }

                // Manual IoU inline for speed
                float inter_x_min = std::max(boxes[idx_i].x_min, boxes[idx_j].x_min);
                float inter_y_min = std::max(boxes[idx_i].y_min, boxes[idx_j].y_min);
                float inter_x_max = std::min(boxes[idx_i].x_max, boxes[idx_j].x_max);
                float inter_y_max = std::min(boxes[idx_i].y_max, boxes[idx_j].y_max);

                float inter_w = std::max(0.0f, inter_x_max - inter_x_min);
                float inter_h = std::max(0.0f, inter_y_max - inter_y_min);
                float inter_area = inter_w * inter_h;

                if (inter_area <= 0)
                    continue;

                float iou = inter_area / (areas[idx_i] + areas[idx_j] - inter_area);

                if (iou > iou_thres) {
                    suppressed[idx_j] = 1;
                }
            }
        }

        return keep;
    }

    void draw_bbox(cv::Mat& image, const BBox& bbox) {

        int x_min = (int)bbox.x_min;
        int y_min = (int)bbox.y_min;
        int x_max = (int)bbox.x_max;
        int y_max = (int)bbox.y_max;
        int cls_id = bbox.cls_id;
        float conf = bbox.conf;
        cv::Scalar box_color = get_color(BOX_COLORS, cls_id);
        cv::Scalar text_color = get_color(TEXT_COLORS, cls_id);

        double font_scale = ((double)image.rows / 640.0);
        double bbox_thickness = font_scale * 3;
        double font_thickness = font_scale * 2;
        cv::Size text_size;
        int baseline;
        char text[64];

        /* bounding box rectangle line */
        cv::rectangle(image,
                      cv::Point(x_min, y_min) /*top left*/,
                      cv::Point(x_max, y_max) /*bottom right*/,
                      box_color,
                      bbox_thickness,
                      cv::LINE_4);

        snprintf(text, 63, "%s(%.f%%)", bbox.cls_name.c_str(), 100 * conf);

        text_size = cv::getTextSize(text, FONT, 2 * font_scale, bbox_thickness, &baseline);

        /* label background rectangle */
        cv::rectangle(image,
                      cv::Rect(x_min,
                               std::max(0, y_min - text_size.height),
                               text_size.width * 0.5,
                               text_size.height),  // top left, width, height
                      box_color,
                      cv::FILLED);

        /* label text */
        cv::putText(image,
                    text,
                    cv::Point(x_min,
                              std::max(0, y_min - text_size.height) == 0
                                      ? text_size.height - 5
                                      : y_min - 10 * font_scale),  // bottom left
                    FONT,
                    font_scale,
                    text_color,
                    font_thickness,
                    cv::LINE_4);
    }

    LetterboxParams compute_letterbox(int ori_w, int ori_h, int model_w, int model_h) {

        // Required parameters
        if (UNLIKELY(ori_w <= 0)) {
            throw std::invalid_argument("ori_width must be provided for YoloUserConfig.");
        }
        if (UNLIKELY(ori_h <= 0)) {
            throw std::invalid_argument("ori_height must be provided for YoloUserConfig.");
        }

        LetterboxParams p;

        p.ratio = std::min((float)model_w / (float)ori_w, (float)model_h / (float)ori_h);

        p.letterbox_w = (int)std::round(ori_w * p.ratio);
        p.letterbox_h = (int)std::round(ori_h * p.ratio);

        p.letterbox_w = std::min(p.letterbox_w, model_w);
        p.letterbox_h = std::min(p.letterbox_h, model_h);

        const int dw = model_w - p.letterbox_w;
        const int dh = model_h - p.letterbox_h;

        p.pad_left = dw / 2;
        p.pad_right = dw - p.pad_left;
        p.pad_top = dh / 2;
        p.pad_bottom = dh - p.pad_top;

        return p;
    }

    cv::Mat preprocess(const cv::Mat& image,
                       int letterbox_w,
                       int letterbox_h,
                       int pad_left,
                       int pad_top,
                       int pad_right,
                       int pad_bottom) {

        cv::Mat resized;
        cv::resize(image, resized, cv::Size(letterbox_w, letterbox_h), 0, 0, cv::INTER_LINEAR);

        // Apply letterbox pad (black border)
        cv::Mat padded;
        cv::copyMakeBorder(resized,
                           padded,
                           pad_top,
                           pad_bottom,
                           pad_left,
                           pad_right,
                           cv::BORDER_CONSTANT,
                           cv::Scalar(0, 0, 0));

        padded.convertTo(padded, CV_32F, 1.0 / 255.0);
        return padded;
    }

    void draw_mask(cv::Mat& image, const Mask& mask, float alpha) {
        // 1. Create an overlay layer for the semi-transparent mask
        cv::Mat overlay = image.clone();

        // Convert custom points to cv::Point
        std::vector<cv::Point> cv_points;
        for (const auto& pt : mask.xys) {
            cv_points.push_back(cv::Point(pt.x, pt.y));
        }

        // 2. Fill the polygon (Mask)
        std::vector<std::vector<cv::Point>> contours = {cv_points};
        cv::Scalar color = get_color(BOX_COLORS, mask.cls_id);
        cv::fillPoly(overlay, contours, color);

        // Blend the mask into the original image
        cv::addWeighted(overlay, alpha, image, 1.0 - alpha, 0, image);

        // 3. Draw Bounding Box
        cv::Rect rect = cv::boundingRect(cv_points);
        cv::rectangle(image, rect, color, 2);  // Thickness of 2

        // 4. Draw Label Text and Background
        std::string text = mask.cls_name;

        int font_face = cv::FONT_HERSHEY_SIMPLEX;
        double font_scale = 0.5;
        int thickness = 1;
        int baseline = 0;

        // Calculate text size to create a background box
        cv::Size text_size = cv::getTextSize(text, font_face, font_scale, thickness, &baseline);
        cv::Point text_org(rect.x, rect.y - 5);  // Position above the top-left of the bbox

        // Ensure the text doesn't go off the top of the screen
        if (text_org.y < 0)
            text_org.y = text_size.height;

        // Draw filled rectangle for text background
        cv::rectangle(image,
                      cv::Point(text_org.x, text_org.y - text_size.height),
                      cv::Point(text_org.x + text_size.width, text_org.y + baseline),
                      color,
                      -1);

        // Draw white text on top of the colored text background
        cv::putText(image,
                    text,
                    text_org,
                    font_face,
                    font_scale,
                    get_color(TEXT_COLORS, mask.cls_id),
                    thickness);

        // Optional: Draw the contour outline
        cv::polylines(image, contours, true, cv::Scalar(255, 255, 255), 1);
    }

    int get_largest_contour_idx(const std::vector<std::vector<cv::Point>>& contours) {
        double max_area = 0.0;
        int largest_contour_idx = -1;
        for (size_t idx = 0; idx < contours.size(); ++idx) {
            if (contours[idx].size() < 3)
                continue;
            double area = cv::contourArea(contours[idx]);
            if (area > max_area) {
                max_area = area;
                largest_contour_idx = static_cast<int>(idx);
            }
        }
        return largest_contour_idx;
    }

    int get_best_label(float& best_score,
                       float* score_buf,
                       const std::vector<int>& valid_classes,
                       float score_thres) {

        best_score = -std::numeric_limits<float>::infinity();
        int best_label = -1;

        // loop through valid classes only
        for (int label : valid_classes) {

            float score = score_buf[label];
            if (score <= score_thres)
                continue;

            if (score <= best_score)
                continue;

            // update
            best_score = score;
            best_label = label;
        }

        // no best label found if label == -1
        return best_label;
    }

    void get_multilabel_scores(std::vector<std::pair<int, float>>& out_labels,
                               float* score_buf,
                               const std::vector<int>& valid_classes,
                               float score_thres) {

        for (int label : valid_classes) {
            float score = score_buf[label];

            // Keep raw score semantics, same as get_best_label.
            if (score <= score_thres)
                continue;

            out_labels.emplace_back(label, score);
        }
    }

    void get_labels_by_mode(std::vector<std::pair<int, float>>& out_labels,
                            float* score_buf,
                            const std::vector<int>& valid_classes,
                            float score_thres,
                            bool use_multi_label) {
        out_labels.clear();
        
        if (use_multi_label) {
            get_multilabel_scores(out_labels, score_buf, valid_classes, score_thres);
            return;
        }

        float best_score = 0.0f;
        const int best_label =
                get_best_label(best_score, score_buf, valid_classes, score_thres);
        if (best_label != -1) {
            out_labels.emplace_back(best_label, best_score);
        }
    }

    /* DFL (Distribution Focal Loss) Decoding */
    std::array<float, 4>
    dfl(float* coord_buf, int row, int col, int stride, int model_w, int model_h) {
        // 1. DFL (Distribution Focal Loss) Decoding
        // YOLO outputs 4 distances (left, top, right, bottom) as probability distributions.
        // We compute the expected value (weighted sum) for each side.
        float dists[4];  // {left, top, right, bottom}

        for (int side = 0; side < 4; ++side) {
            float* side_dist_buf = coord_buf + side * 16;

            // Numerically stable Softmax: find max first
            float local_max = side_dist_buf[0];

// cool optimization: if possible, use AVX512 for finding the max
#if defined(__AVX512F__)
#pragma omp simd reduction(max : local_max)
            for (int i = 0; i < 16; ++i) {  // re-checking the 0 is to align the simd
                if (side_dist_buf[i] > local_max)
                    local_max = side_dist_buf[i];
            }
#else
            // else the scalar version is faster
            for (int i = 1; i < 16; ++i) {
                if (side_dist_buf[i] > local_max)
                    local_max = side_dist_buf[i];
            }
#endif

            float softmax_sum = 0.0f;
            float weighted_sum = 0.0f;

// Single pass for exp calculation to optimize performance
#pragma omp simd reduction(+ : softmax_sum, weighted_sum)
            for (int i = 0; i < 16; ++i) {
                float exp_val = expf(side_dist_buf[i] - local_max);
                softmax_sum += exp_val;
                weighted_sum += (float)i * exp_val;
            }
            dists[side] = weighted_sum / softmax_sum;
        }

        // 2. Decode Distances to Anchor-Relative Coordinates
        // Coordinates are relative to the grid cell center (row, col) multiplied by stride.
        // d[0]=left, d[1]=top, d[2]=right, d[3]=bottom
        float x1 = (col + 0.5f - dists[0]) * stride;
        float y1 = (row + 0.5f - dists[1]) * stride;
        float x2 = (col + 0.5f + dists[2]) * stride;
        float y2 = (row + 0.5f + dists[3]) * stride;

        // make sure coords are within letterbox size
        x1 = std::clamp(x1, 0.0f, (float)model_w - 1.f);
        y1 = std::clamp(y1, 0.0f, (float)model_h - 1.f);
        x2 = std::clamp(x2, 0.0f, (float)model_w - 1.f);
        y2 = std::clamp(y2, 0.0f, (float)model_h - 1.f);

        // coords for letterbox
        return {x1, y1, x2, y2};
    }

    std::string normalize(std::string s) {
        for (char& c : s) {
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }

        // also replace any _ with -
        std::replace(s.begin(), s.end(), '_', '-');

        return s;
    }

    // Levenshtein distance implementation for string similarity
    std::size_t levenshtein(const std::string& a, const std::string& b) {
        const std::size_t n = a.size(), m = b.size();
        if (n == 0)
            return m;
        if (m == 0)
            return n;

        std::vector<std::size_t> prev(m + 1), cur(m + 1);
        for (std::size_t j = 0; j <= m; ++j)
            prev[j] = j;

        for (std::size_t i = 1; i <= n; ++i) {
            cur[0] = i;
            for (std::size_t j = 1; j <= m; ++j) {
                const std::size_t cost = (a[i - 1] == b[j - 1]) ? 0 : 1;
                cur[j] = std::min({prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost});
            }
            std::swap(prev, cur);
        }
        return prev[m];
    }
    std::string colored_diff(const std::string& input, const std::string& target) {
        const size_t n = input.size();
        const size_t m = target.size();

        // Build DP table
        std::vector<std::vector<size_t>> dp(n + 1, std::vector<size_t>(m + 1));

        for (size_t i = 0; i <= n; ++i)
            dp[i][0] = i;
        for (size_t j = 0; j <= m; ++j)
            dp[0][j] = j;

        for (size_t i = 1; i <= n; ++i) {
            for (size_t j = 1; j <= m; ++j) {
                size_t cost = (input[i - 1] == target[j - 1]) ? 0 : 1;
                dp[i][j] = std::min({
                        dp[i - 1][j] + 1,        // deletion
                        dp[i][j - 1] + 1,        // insertion
                        dp[i - 1][j - 1] + cost  // substitution
                });
            }
        }

        // Backtrack to build colored output
        std::string result;
        size_t i = n, j = m;

        while (i > 0 || j > 0) {
            if (i > 0 && j > 0 && dp[i][j] == dp[i - 1][j - 1] && input[i - 1] == target[j - 1]) {
                // Match
                result = input[i - 1] + result;
                --i;
                --j;
            } else if (i > 0 && j > 0 && dp[i][j] == dp[i - 1][j - 1] + 1) {
                // Substitution
                result = std::string(termcolor::green) + target[j - 1] + termcolor::reset + result;
                --i;
                --j;
            } else if (j > 0 && dp[i][j] == dp[i][j - 1] + 1) {
                // Insertion (missing char)
                result = std::string(termcolor::green) + target[j - 1] + termcolor::reset + result;
                --j;
            } else {
                // Deletion (extra char)
                result = std::string(termcolor::red) + input[i - 1] + termcolor::reset + result;
                --i;
            }
        }

        return result;
    }

}
