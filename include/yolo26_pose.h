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
#include "MxPrepost.h"
#include "utils.h"

#include <memx/accl/utils/rental_store.h>
#include <memory>
#include <vector>

namespace MX {
    namespace Prepost {
        class Yolo26Pose : public MX::Prepost::MxPrepost {
          // Color list for drawing keypoints
        inline static const std::vector<cv::Scalar> YOLO26_KEYPOINT_COLORS = {
          cv::Scalar(255, 0, 0),   cv::Scalar(0, 255, 0),  cv::Scalar(0, 0, 255),
          cv::Scalar(255, 255, 0),   cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 255),
          cv::Scalar(128, 128, 128),  cv::Scalar(128, 0, 128), cv::Scalar(0, 128, 128),
          cv::Scalar(128, 128, 0), cv::Scalar(0, 128, 0),   cv::Scalar(128, 0, 0),
          cv::Scalar(0, 0, 128),       cv::Scalar(0, 0, 0),   cv::Scalar(0, 0, 0),
          cv::Scalar(0, 0, 0),   cv::Scalar(0, 0, 0),   cv::Scalar(0, 0, 0),
          cv::Scalar(0, 0, 0),   cv::Scalar(0, 0, 0),   cv::Scalar(0, 0, 0)};

          public:
            Yolo26Pose(MX::Runtime::MxAcclBase* accl,
                                const YoloUserConfig& config,
                                const std::string& task = "");

            cv::Mat preprocess(const cv::Mat& image) override;
            void postprocess(const std::vector<float*>& outputs, Result& result) override;
            void postprocess(const std::vector<float*>& outputs,
                            Result& result,
                            const cv::Mat& original_image);
            void
            postprocess(const std::vector<float*>& outputs, Result& result, int ori_h, int ori_w);
            void draw(cv::Mat& image, const Result& result) override;

          private:
            //-----------------------------
            // model/task-specific constants
            static constexpr int    COORD_FMAP_SIZE = 4; // number of channels in the coordinate ofmap
            static constexpr std::array<int, 3> STRIDES = {8, 16, 32}; // fixed by model arch
            static constexpr int    NUM_LAYERS = STRIDES.size();
            static constexpr int    NUM_KEYPOINTS = 17;   // Number of keypoints for pose estimation
            //-----------------------------
            // Pairs of keypoints for drawing skeleton
            static constexpr std::array<const std::pair<int, int>, 18> KEYPOINT_PAIRS = {{
                    {0, 1},
                    {0, 2},
                    {1, 3},
                    {2, 4},
                    {0, 5},
                    {0, 6},
                    {5, 7},
                    {7, 9},
                    {6, 8},
                    {8, 10},
                    {5, 6},
                    {5, 11},
                    {6, 12},
                    {11, 12},
                    {11, 13},
                    {13, 15},
                    {12, 14},
                    {14, 16},
            }};
            //-----------------------------
            size_t total_preds_ = 0;
            MX::Utils::RentalStore<Util::PoseScratch> pose_scratch_pool_{rentalStoreSlots};

            void postprocess_impl(const std::vector<float*>& outputs,
                                  Result& result,
                                  int ori_h,
                                  int ori_w);
            std::vector<MX::Prepost::Util::LayerParams> yolo_post_layers_;

            std::unique_ptr<MX::Prepost::Util::ScoreManager> smgr_;
            YoloFinalConfig cfg_;
        };

    }
}
