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

// forward declaration
namespace MX::Prepost::Util {
    class ScoreManager;
}

namespace MX {
    namespace Prepost {
        class YoloUltralyticsSegment : public MX::Prepost::MxPrepost {

          public:
            YoloUltralyticsSegment(MX::Runtime::MxAcclBase* accl,
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
            static constexpr int    COORD_FMAP_SIZE = 64; // number of channels in the coordinate ofmap
            static constexpr std::array<int, 3> STRIDES = {8, 16, 32}; // fixed by model arch
            static constexpr int    NUM_LAYERS = STRIDES.size();
            static constexpr int    MASK_DIV_FACTOR = 4; // YOLO seg masks are usually 1/4 the input size
            static constexpr int    MASK_CHANNELS = 32; // used for mask coef and mask proto fmaps
            static constexpr float MASK_THRESH = 0.0f;
            //-----------------------------
            size_t total_preds_ = 0;
            MX::Utils::RentalStore<Util::SegmentScratch> segment_scratch_pool_{rentalStoreSlots};
            MX::Utils::RentalStore<std::vector<std::pair<int, float>>> label_pool_{rentalStoreSlots};

            void postprocess_impl(const std::vector<float*>& outputs,
                                  Result& result,
                                  int ori_h,
                                  int ori_w);
            std::vector<MX::Prepost::Util::LayerParams> yolo_post_layers_;


            std::unique_ptr<MX::Prepost::Util::ScoreManager> smgr_;
            YoloFinalConfig cfg_;
            uint8_t mask_proto_port_;
            int mask_proto_w_;
            int mask_proto_h_;
            float inv_mask_thresh_;
        };

    }
}
