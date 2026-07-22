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
#include <memx/accl/MxAccl.h>
#include <memx/accl/MxAcclBase.h>

namespace MX {
    namespace Prepost {
        class MxError : public std::runtime_error {
          public:
            using std::runtime_error::runtime_error;
        };

        class UnsupportedTaskError : public MxError {
          public:
            using MxError::MxError;
        };


        /**
         * Class for MX pre/post-processing. This class defines the interface for preprocessing input images, postprocessing model outputs, and drawing results on images.
         *
         * You should use the factory functions to create task-specific pre/post-processing objects, and assign to a pointer of this base class type. For example:
         * ```cpp
         * std::unique_ptr<MxPrepost> prepost = std::make_unique<MxPrepost>();
         * try {
         *    prepost.reset(MxPrepost::create(accl, "yolov8-det", config));
         * } catch (const UnsupportedTaskError& e) {
         *    std::cerr << "Error creating MxPrepost: " << e.what() << std::endl;
         * }
         * ```
         *
         * Or using the safe factory function:
         * ```cpp
         * std::unique_ptr<MxPrepost> prepost;
         * std::string err;
         * if (!MxPrepost::create_safe(accl, "yolov8-det", config, prepost, err)) {
         *   std::cerr << "Error creating MxPrepost: " << err << std::endl;
         * } else {
         *   // prepost is successfully created and can be used
         * }
         * ```
         *
         */
        class MxPrepost {
          public:
            virtual ~MxPrepost() = default;

            /**
             * Preprocess input image for model inference.
             *
             * @param input Input image as cv::Mat (in RGB format).
             *
             * @return Preprocessed image as cv::Mat, ready to be fed into the MXA.
             */
            virtual cv::Mat preprocess(const cv::Mat& input) = 0;

            // Legacy API (keep for compatibility; derived YOLO classes can throw here)
            virtual void postprocess(const std::vector<float*>& outputs, Result& result) = 0;

            /**
             * Postprocess the output feature maps from the MXA and use original image dimensions for scaling.
             *
             * @param outputs  Vector of pointers to float arrays, each being an output feature map from the MXA (and after a call to FeatureMap::get_data()).
             * @param result   Output Result object to be filled with post-processing results (e.g., detected boxes, masks, keypoints).
             * @param ori_w    Original width of the input image before preprocessing (used for scaling post-processing results back to original image space).
             * @param ori_h    Original height of the input image before preprocessing (used for scaling post-processing results back to original image space).
             */
            virtual void postprocess(const std::vector<float*>& outputs,
                                     Result& result,
                                     int ori_w,
                                     int ori_h) = 0;

            /**
             * Postporcess the output feature maps from the MXA and get the original image's dimensions from the original image itself (instead of passing ori_w and ori_h separately).
             *
             * @param outputs  Vector of pointers to float arrays, each being an output feature map from the MXA (and after a call to FeatureMap::get_data()).
             * @param result   Output Result object to be filled with post-processing results (e.g., detected boxes, masks, keypoints).
             * @param original_image The original input image as a cv::Mat (in RGB format). This is used to obtain the original dimensions for scaling post-processing results back to original image space.
             */
            virtual void postprocess(const std::vector<float*>& outputs,
                                     Result& result,
                                     const cv::Mat& original_image) = 0;

            /**
             * Draw the post-processing Result data on the given image (e.g., draw detected boxes, masks, keypoints, etc.).
             *
             * @param image  The image (in BGR format) on which to draw the post-processing results. This **will be modified in-place**.
             * @param result The Result object containing post-processing results to be drawn on the image.
             */
            virtual void draw(cv::Mat& image, const Result& result) = 0;

            /**
             * Factory function to create a pre/post-processing object for the given task.
             *
             * @param accl   Accelerator runtime object: MxAccl or MxAcclMT
             * @param task   Task string like "yolov8-det".
             * @param config User config.
             *
             * @return Raw pointer owned by caller (or wrap into std::unique_ptr).
             *
             * @throws UnsupportedTaskError if task is not recognized.
             */
            static MxPrepost* create(MX::Runtime::MxAcclBase* accl,
                                     const std::string& task,
                                     const YoloUserConfig& config);

            /**
             * No-throw factory function to create a pre/post-processing object for the given task.
             *
             * @param accl   Accelerator runtime object: MxAccl or MxAcclMT
             * @param task   Task string like "yolov8-det".
             * @param config User config.
             * @param out    Output unique_ptr to hold the created MxPrepost object if successful.
             * @param err    Output string to hold error message if creation fails.
             *
             * @return true if creation is successful, false otherwise. If false is returned, 'out' will be set to nullptr and 'err' will contain the error message.
             */
            static bool create_safe(MX::Runtime::MxAcclBase* accl,
                                    const std::string& task,
                                    const YoloUserConfig& config,
                                    std::unique_ptr<MxPrepost>& out,
                                    std::string& err) noexcept;
        };

    }  // namespace Prepost
}  // namespace MX
