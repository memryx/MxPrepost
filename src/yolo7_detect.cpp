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
#include <memx/accl/utils/macros.h>
#include <memx/accl/MxAcclBase.h>

#include "yolo7_detect.h"

#include "config_finalizer.h"
#include "utils.h"

using namespace MX::Runtime;
using namespace MX::Prepost;
using namespace MX::Prepost::Util;

Yolo7Detect::Yolo7Detect(MX::Runtime::MxAcclBase* accl,
                         const YoloUserConfig& user_cfg,
                         const std::string& task) {

    // init settings from config
    cfg_ = ConfigFinalizer::finalize(accl, user_cfg);

    // init score manager
    smgr_ = std::make_unique<MX::Prepost::Util::ScoreManager>(cfg_.conf, cfg_.fast_sigmoid);

    auto model_info = accl->get_model_info(cfg_.model_id);

    yolo_post_layers_.resize(NUM_LAYERS);

    // total_preds_ is the sum of (height * width) across all layers
    total_preds_ = 0;
    for(int i=0; i < NUM_LAYERS; ++i) {
        int stride = STRIDES[i];
        total_preds_ += (cfg_.model_h / stride) * (cfg_.model_w / stride);
    }

    if(cfg_.override_layer_mapping.empty()){
        // YOLOv7 has a single output per layer with both coord and conf, so we just need to find the port that matches the expected shape for each layer
        for(int i=0; i < NUM_LAYERS; ++i) {
            int stride = STRIDES[i];
            yolo_post_layers_[i].height = cfg_.model_h / stride;
            yolo_post_layers_[i].width = cfg_.model_w / stride;
            yolo_post_layers_[i].stride = stride;

            MX::Types::ShapeVector expected_shape{cfg_.model_h / stride, cfg_.model_w / stride, 1, static_cast<int64_t>(3 * (5 + cfg_.class_labels.size()))};

            bool found_port = false;
            for (size_t port = 0; port < model_info.output_layer_names.size(); ++port) {
                const auto& actual_shape = model_info.out_featuremap_shapes[port];
                if (actual_shape == expected_shape) {
                    yolo_post_layers_[i].port_out = port;
                    found_port = true;
                    break;
                }
            }

            if(!found_port) {
                std::string error_message = "Could not find output port for YOLOv7 layer with stride " + std::to_string(stride) + ". Expected output shape: " + expected_shape.to_string() + ". Please provide an explicit mapping using the override_layer_mapping entry in YoloUserConfig.";
                throw std::runtime_error(error_message);
            }
        }
    } 
    else {
        // take the user-provided mapping, but verify the shapes match else error
        int num_found_layers = 0;
        for (const auto& [stride, layer_names] : cfg_.override_layer_mapping) {
            int layer_id = -1;
            for (int i=0; i < NUM_LAYERS; ++i) {
                if (STRIDES[i] == stride) {
                    layer_id = i;
                    break;
                }
            }

            // invalid stride
            if (layer_id == -1) {
                std::string error_message = "Invalid stride " + std::to_string(stride) + " in override_layer_mapping. Accepted STRIDES are: ";
                for (size_t i = 0; i < STRIDES.size(); ++i) {
                    error_message += std::to_string(STRIDES[i]);
                    if (i != STRIDES.size() - 1) {
                        error_message += ", ";
                    }
                }
                throw std::runtime_error(error_message);
            }

            bool found_port = false;
            MX::Types::ShapeVector expected_shape{cfg_.model_h / stride, cfg_.model_w / stride, 1, static_cast<int64_t>(3 * (5 + cfg_.class_labels.size()))};
            yolo_post_layers_[layer_id].height = cfg_.model_h / stride;
            yolo_post_layers_[layer_id].width = cfg_.model_w / stride;
            yolo_post_layers_[layer_id].stride = stride;

            for (size_t port = 0; port < model_info.output_layer_names.size(); ++port) {
                const auto& actual_shape = model_info.out_featuremap_shapes[port];
                // find the output port by name AND matching shape
                if (model_info.output_layer_names[port] == layer_names[0]) {
                    if (actual_shape == expected_shape) {
                        yolo_post_layers_[layer_id].port_out = port;
                        found_port = true;
                        break;
                    } else {
                        std::string error_message = "Output port " + std::to_string(port) + " with name " + layer_names[0] + " has shape " 
                            + actual_shape.to_string() + " which does not match expected shape for YOLOv7 layer with stride " + std::to_string(stride) + " which is "
                            + expected_shape.to_string() + ". Please check your override_layer_mapping entry in YoloUserConfig.";
                        throw std::runtime_error(error_message);
                    }
                }
            }

            if(!found_port) {
                std::string error_message = "Could not find output port for YOLOv7 layer with stride " + std::to_string(stride) + ". Please check your override_layer_mapping entry in YoloUserConfig.";
                throw std::runtime_error(error_message);
            }

            num_found_layers++;
        }

        if(num_found_layers != NUM_LAYERS) {
            throw std::runtime_error("override_layer_mapping must contain entries for all " + std::to_string(NUM_LAYERS) + " layers. Found entries for " + std::to_string(num_found_layers) + " layers.");
        }
    }

    for (unsigned i = 0; i < rentalStoreSlots; ++i) {
        MX::Prepost::Util::BBoxScratch* slot = bbox_pool_.checkout_item();
        if (slot == nullptr) {
            throw std::runtime_error("Yolo7Detect: bbox pool warmup failed");
        }
        slot->boxes.reserve(total_preds_);
        bbox_pool_.return_item(slot);
    }
}

cv::Mat Yolo7Detect::preprocess(const cv::Mat& image) {
    const int ori_w = image.cols;
    const int ori_h = image.rows;

    const auto lb = compute_letterbox(ori_w, ori_h, cfg_.model_w, cfg_.model_h);

    return MX::Prepost::Util::preprocess(image,
                                         lb.letterbox_w,
                                         lb.letterbox_h,
                                         lb.pad_left,
                                         lb.pad_top,
                                         lb.pad_right,
                                         lb.pad_bottom);
}

void Yolo7Detect::draw(cv::Mat& image, const Result& result) {
    for (const BBox& bbox : result.boxes) {
        MX::Prepost::Util::draw_bbox(image, bbox);
    }
}

void Yolo7Detect::postprocess(const std::vector<float*>&, Result&) {
    throw std::runtime_error(
            "postprocess(outputs, result) requires original image or (ori_w, ori_h). "
            "Use postprocess(outputs, result, original_image) or postprocess(outputs, result, ori_w, ori_h).");
}

void Yolo7Detect::postprocess(const std::vector<float*>& outputs,
                              Result& result,
                              const cv::Mat& original_image) {
    if (UNLIKELY(original_image.empty())) {
        throw std::invalid_argument("original_image must be non-empty for postprocess");
    }
    postprocess_impl(outputs, result, original_image.rows, original_image.cols);
}

void Yolo7Detect::postprocess(const std::vector<float*>& outputs,
                              Result& result,
                              int ori_h,
                              int ori_w) {
    if (UNLIKELY(ori_w <= 0 || ori_h <= 0)) {
        throw std::invalid_argument("ori_w and ori_h must be > 0 for postprocess");
    }
    postprocess_impl(outputs, result, ori_h, ori_w);
}

void Yolo7Detect::postprocess_impl(const std::vector<float*>& outputs,
                                   Result& result,
                                   int ori_h,
                                   int ori_w) {

    result.boxes.clear();
    const auto lb = compute_letterbox(ori_w, ori_h, cfg_.model_w, cfg_.model_h);

    auto bbox_guard = MX::Prepost::Util::make_rental_pool_guard(
            bbox_pool_, "Yolo7Detect: bbox pool checkout failed");
    std::vector<BBox>& all_boxes = bbox_guard.ptr->boxes;
    all_boxes.clear();

    const int num_classes_total = static_cast<int>(cfg_.class_labels.size());

    const int kNumAnchors = 3;
    const int per_anchor = 5 + num_classes_total;  // [tx,ty,tw,th,obj] + classes
    const int per_cell = kNumAnchors * per_anchor;

    for (size_t layer_id = 0; layer_id < NUM_LAYERS; ++layer_id) {
        const auto& layer = yolo_post_layers_[layer_id];
        float* out_base = outputs.at(layer.port_out);

        for (size_t i = 0; i < layer.height * layer.width; ++i) {

            float* cell = out_base + i * per_cell;

            int best_label = -1;
            int best_anchor = -1;
            float best_score = 0.0f;

            for (int a = 0; a < 3; ++a) {
                float* p = cell + a * per_anchor;

                const float obj_logit = p[4];

                float cls_logit = 0;
                int label = MX::Prepost::Util::get_best_label(
                        cls_logit, p + 5, cfg_.valid_classes, -1e9f);

                if (label == -1)
                    continue;

                const float obj_prob = smgr_->convert(obj_logit);
                const float cls_prob = smgr_->convert(cls_logit);
                const float score = obj_prob * cls_prob;

                if (best_label == -1 || score > best_score) {
                    best_label = label;
                    best_anchor = a;
                    best_score = score;
                }
            }

            if (best_label == -1)
                continue;

            if (best_score < cfg_.conf)
                continue;

            // Decode bbox for chosen anchor
            float* p = cell + best_anchor * per_anchor;

            const float tx = p[0];
            const float ty = p[1];
            const float tw = p[2];
            const float th = p[3];

            const float sx = smgr_->convert(tx);
            const float sy = smgr_->convert(ty);
            const float sw = smgr_->convert(tw);
            const float sh = smgr_->convert(th);

            const float stride = static_cast<float>(layer.stride);
            const float row = static_cast<float>(i / layer.width);
            const float col = static_cast<float>(i % layer.width);

            static const float anchors_w[3][3] = {
                    {12.f, 19.f, 40.f},     // stride 8
                    {36.f, 76.f, 72.f},     // stride 16
                    {142.f, 192.f, 459.f},  // stride 32
            };
            static const float anchors_h[3][3] = {
                    {16.f, 36.f, 28.f},
                    {75.f, 55.f, 146.f},
                    {110.f, 243.f, 401.f},
            };

            const float cx = (sx * 2.0f - 0.5f + col) * stride;
            const float cy = (sy * 2.0f - 0.5f + row) * stride;

            float bw = (sw * 2.0f);
            float bh = (sh * 2.0f);
            bw = bw * bw * anchors_w[layer_id][best_anchor];
            bh = bh * bh * anchors_h[layer_id][best_anchor];

            std::array<float, 4> coord = {
                    cx - bw * 0.5f, cy - bh * 0.5f, cx + bw * 0.5f, cy + bh * 0.5f};

            // Undo letterbox using per-call padding/ratio (IMPORTANT: left for x, top for y)
            coord[0] = (coord[0] - lb.pad_left) / lb.ratio;
            coord[1] = (coord[1] - lb.pad_top) / lb.ratio;
            coord[2] = (coord[2] - lb.pad_left) / lb.ratio;
            coord[3] = (coord[3] - lb.pad_top) / lb.ratio;
            // Clip to image bounds
            coord[0] = std::max(0.0f, std::min(coord[0], (float)ori_w));
            coord[1] = std::max(0.0f, std::min(coord[1], (float)ori_h));
            coord[2] = std::max(0.0f, std::min(coord[2], (float)ori_w));
            coord[3] = std::max(0.0f, std::min(coord[3], (float)ori_h));

            // Drop invalid/degenerate boxes
            if (coord[2] <= coord[0] || coord[3] <= coord[1]) {
                continue;
            }
            // store bbox
            all_boxes.emplace_back(coord[0],
                                   coord[1],
                                   coord[2],
                                   coord[3],
                                   best_score,
                                   best_label,
                                   cfg_.class_labels[best_label]);
        }
    }
    // apply NMS
    std::vector<int> keep_indices =
            MX::Prepost::Util::nms(all_boxes, cfg_.iou, cfg_.class_agnostic, cfg_.max_dets);

    if (keep_indices.empty())
        return;

    result.boxes.reserve(keep_indices.size());
    for (int idx : keep_indices) {
        result.boxes.push_back(all_boxes[idx]);
    }
}
