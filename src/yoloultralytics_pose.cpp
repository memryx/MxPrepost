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

#include "yoloultralytics_pose.h"

#include "config_finalizer.h"
#include "utils.h"

using namespace MX::Prepost;
using namespace MX::Prepost::Util;

YoloUltralyticsPose::YoloUltralyticsPose(MX::Runtime::MxAcclBase* accl,
                                         const YoloUserConfig& user_cfg,
                                         const std::string& task) {

    // if user_cfg.classmap_path + user_cfg.custom_class_labels are both empty, don't let
    // ConfigFinalizer::finalize init with the 80 COCO classes. Instead, use
    // one class with label "person".
    if(user_cfg.classmap_path.empty() && user_cfg.custom_class_labels.empty()){
        YoloUserConfig modified_cfg = user_cfg;
        modified_cfg.custom_class_labels = {"person"};
        cfg_ = ConfigFinalizer::finalize(accl, modified_cfg);
    }
    else {
        cfg_ = ConfigFinalizer::finalize(accl, user_cfg);
    }

    // init score manager
    smgr_ = std::make_unique<MX::Prepost::Util::ScoreManager>(cfg_.conf, cfg_.fast_sigmoid);

    // autocalculate the layer shapes and corresponding ports based on model output shapes
    // and expected shapes, unless user provided an explicit mapping in the config (override_layer_mapping)
    auto model_info = accl->get_model_info(cfg_.model_id);

    yolo_post_layers_.resize(NUM_LAYERS);

    std::vector<MX::Types::ShapeVector> expected_coord_shapes;
    std::vector<MX::Types::ShapeVector> expected_conf_shapes;
    std::vector<MX::Types::ShapeVector> expected_keypt_shapes;

    total_preds_ = 0;

    // expected coord/conf are the same as det
    for(int i=0; i < NUM_LAYERS; ++i) {
        int stride = STRIDES[i];
        expected_coord_shapes.push_back({cfg_.model_h / stride, cfg_.model_w / stride, 1, COORD_FMAP_SIZE});
        expected_conf_shapes.push_back({cfg_.model_h / stride, cfg_.model_w / stride, 1, static_cast<int64_t>(cfg_.class_labels.size())});
        expected_keypt_shapes.push_back({cfg_.model_h / stride, cfg_.model_w / stride, 1, NUM_KEYPOINTS * 3}); // each keypoint has (x, y, conf)
        total_preds_ += (cfg_.model_h / stride) * (cfg_.model_w / stride);
    }

    int num_ofmaps = model_info.out_featuremap_shapes.size();

    // autodetect
    if(cfg_.override_layer_mapping.empty()){

        // match expected shapes with actual model input shapes to find the correct ports for each layer
        if (num_ofmaps != NUM_LAYERS * 3) {
            throw std::runtime_error("Expected " + std::to_string(NUM_LAYERS * 3) + " output feature maps (coord, conf, and keypt for each stride layer), but model has " + std::to_string(num_ofmaps) + ". Please check your model and config.");
        }

        if(cfg_.class_labels.size() == COORD_FMAP_SIZE || cfg_.class_labels.size() == NUM_KEYPOINTS * 3){
            // error: what the heck is going on
            throw std::runtime_error("... why does your pose model have " + std::to_string(cfg_.class_labels.size()) + " classes? Please check your config!");
        }
        else {
            for(int i=0; i < NUM_LAYERS; ++i){
                int stride = STRIDES[i];
                yolo_post_layers_[i].height = cfg_.model_h / stride;
                yolo_post_layers_[i].width = cfg_.model_w / stride;
                yolo_post_layers_[i].stride = stride;

                // populate anchor grid
                for(unsigned int y=0; y < yolo_post_layers_[i].height; ++y){
                    for(unsigned int x=0; x < yolo_post_layers_[i].width; ++x){
                        yolo_post_layers_[i].anchors.push_back({x + 0.5f, y + 0.5f}); // center
                    }
                }

                bool found_coord = false;
                bool found_conf  = false;
                bool fount_keypt = false;
                for (int port= 0; port < num_ofmaps; ++port) {
                    const auto& actual_shape = model_info.out_featuremap_shapes[port];
                    if (actual_shape == expected_coord_shapes[i]) {
                        yolo_post_layers_[i].coord_port = port;
                        found_coord = true;
                    } else if (actual_shape == expected_conf_shapes[i]) {
                        yolo_post_layers_[i].conf_port = port;
                        found_conf = true;
                    } else if (actual_shape == expected_keypt_shapes[i]) {
                        yolo_post_layers_[i].keypt_port = port;
                        fount_keypt = true;
                    }
                }

                if (!found_coord) {
                    throw std::runtime_error("Could not find coordinate output port for layer with stride " + std::to_string(stride) + ". Please check your model and config.");
                }
                if (!found_conf) {
                    throw std::runtime_error("Could not find confidence output port for layer with stride " + std::to_string(stride) + ". Please check your model and config.");
                }
                if (!fount_keypt) {
                    throw std::runtime_error("Could not find keypoint output port for layer with stride " + std::to_string(stride) + ". Please check your model and config.");
                }
            }

        }



    }
    else {
        // user-provided mapping. Validate that all specified ports and shapes are correct.
        int num_found_layers = 0;

        // override_layer_mapping has 3 layer_names: coord, conf, and keypt
        for (const auto& [stride, layer_names] : cfg_.override_layer_mapping) {
            int layer_id = -1;
            // find the layer_id corresponding to this stride
            for (int i = 0; i < (int) STRIDES.size(); ++i) {
                if (STRIDES[i] == stride) {
                    layer_id = i;
                    break;
                }
            }

            // invalid stride
            if(layer_id == -1){
                std::string error_message = "Invalid stride " + std::to_string(stride) + " in override_layer_mapping. Accepted STRIDES are: ";
                for (size_t i = 0; i < STRIDES.size(); ++i) {
                    error_message += std::to_string(STRIDES[i]);
                    if (i != STRIDES.size() - 1) {
                        error_message += ", ";
                    }
                }
                throw std::runtime_error(error_message);
            }

            
            bool found_coord = false;
            bool found_conf = false;
            bool found_keypt = false;
            for (int port = 0; port < num_ofmaps; ++port) {
                const auto& actual_shape = model_info.out_featuremap_shapes[port];
                if (model_info.output_layer_names[port] == layer_names[0]) { // coord
                    if (actual_shape == expected_coord_shapes[layer_id]) {
                        yolo_post_layers_[layer_id].coord_port = port;
                        found_coord = true;
                    } else {
                        std::string error_message = "Output port " + std::to_string(port) + " with name " + layer_names[0] + " has shape "
                            + actual_shape.to_string() + " which does not match expected coordinate shape for stride " + std::to_string(stride) + " which is "
                            + expected_coord_shapes[layer_id].to_string();
                        throw std::runtime_error(error_message);
                    }
                } else if (model_info.output_layer_names[port] == layer_names[1]) { // conf
                    if (actual_shape == expected_conf_shapes[layer_id]) {
                        yolo_post_layers_[layer_id].conf_port = port;
                        found_conf = true;
                    } else {
                        std::string error_message = "Output port " + std::to_string(port) + " with name " + layer_names[1] + " has shape "
                            + actual_shape.to_string() + " which does not match expected confidence shape for stride " + std::to_string(stride) + " which is "
                            + expected_conf_shapes[layer_id].to_string();
                        throw std::runtime_error(error_message);
                    }
                } else if (model_info.output_layer_names[port] == layer_names[2]) { // keypt
                    if (actual_shape == expected_keypt_shapes[layer_id]) {
                        yolo_post_layers_[layer_id].keypt_port = port;
                        found_keypt = true;
                    } else {
                        std::string error_message = "Output port " + std::to_string(port) + " with name " + layer_names[2] + " has shape "
                            + actual_shape.to_string() + " which does not match expected keypoint shape for stride " + std::to_string(stride) + " which is "
                            + expected_keypt_shapes[layer_id].to_string();
                        throw std::runtime_error(error_message);
                    }
                }
            }

            if(!found_coord) {
                throw std::runtime_error("Could not find coordinate output port for layer with stride " + std::to_string(stride) + ". Please check your override_layer_mapping entry in YoloUserConfig.");
            }
            if(!found_conf) {
                throw std::runtime_error("Could not find confidence output port for layer with stride " + std::to_string(stride) + ". Please check your override_layer_mapping entry in YoloUserConfig.");
            }
            if(!found_keypt) {
                throw std::runtime_error("Could not find keypoint output port for layer with stride " + std::to_string(stride) + ". Please check your override_layer_mapping entry in YoloUserConfig.");
            }

        } // for each stride in override_layer_mapping
  
        if(num_found_layers != NUM_LAYERS * 3) {
            std::string error_message = "override_layer_mapping is missing some layers. Expected " + std::to_string(NUM_LAYERS * 3) + " layers (coord, conf, and keypt for each stride layer), but found " + std::to_string(num_found_layers) + ". Please check your override_layer_mapping entry in YoloUserConfig.";
            throw std::runtime_error(error_message);
        }

    } // if override_layer_mapping provided

    for (unsigned i = 0; i < rentalStoreSlots; ++i) {
        Util::PoseScratch* slot = pose_scratch_pool_.checkout_item();
        if (slot == nullptr) {
            throw std::runtime_error("YoloUltralyticsPose: pose rental store warmup failed");
        }
        slot->boxes.reserve(total_preds_);
        slot->keypoints.reserve(total_preds_);
        pose_scratch_pool_.return_item(slot);
    }
}

cv::Mat YoloUltralyticsPose::preprocess(const cv::Mat& image) {
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

void YoloUltralyticsPose::draw(cv::Mat& image, const Result& result) {

    // draw bbox
    for (const BBox& bbox : result.boxes) {
        MX::Prepost::Util::draw_bbox(image, bbox);
    }

    // draw keypoints and skeleton
    for (const auto& kpts_per_box : result.keypoints) {

        // draw lines (skeletons)
        for (const auto& connection : KEYPOINT_PAIRS) {
            unsigned int idx1 = connection.first;
            unsigned int idx2 = connection.second;

            if (idx1 < kpts_per_box.size() && idx2 < kpts_per_box.size()) {
                auto kpt1 = kpts_per_box[idx1];
                auto kpt2 = kpts_per_box[idx2];

                if (kpt1.xy.x != -1 && kpt2.xy.x != -1) {
                    cv::line(image,
                             cv::Point(kpt1.xy.x, kpt1.xy.y),
                             cv::Point(kpt2.xy.x, kpt2.xy.y),
                             cv::Scalar(255, 255, 255),
                             3);
                }
            }
        }

        // Draw individual keypoints
        for (unsigned int i = 0; i < kpts_per_box.size(); ++i) {
            auto& kpt = kpts_per_box[i];
            if (kpt.xy.x != -1) {
                cv::circle(image,
                           cv::Point(kpt.xy.x, kpt.xy.y),
                           4,
                           KEYPOINT_COLORS[i % KEYPOINT_COLORS.size()],
                           -1);
            }
        }
    }
}

void YoloUltralyticsPose::postprocess(const std::vector<float*>&, Result&) {
    throw std::runtime_error(
            "postprocess(outputs, result) requires original image or (ori_w, ori_h). "
            "Use postprocess(outputs, result, original_image) or postprocess(outputs, result, ori_w, ori_h).");
}

void YoloUltralyticsPose::postprocess(const std::vector<float*>& outputs,
                                      Result& result,
                                      const cv::Mat& original_image) {
    if (UNLIKELY(original_image.empty())) {
        throw std::invalid_argument("original_image must be non-empty for postprocess");
    }
    postprocess_impl(outputs, result, original_image.rows, original_image.cols);
}

void YoloUltralyticsPose::postprocess(const std::vector<float*>& outputs,
                                      Result& result,
                                      int ori_h,
                                      int ori_w) {
    if (UNLIKELY(ori_w <= 0 || ori_h <= 0)) {
        throw std::invalid_argument("ori_w and ori_h must be > 0 for postprocess");
    }
    postprocess_impl(outputs, result, ori_h, ori_w);
}

void YoloUltralyticsPose::postprocess_impl(const std::vector<float*>& outputs,
                                           Result& result,
                                           int ori_h,
                                           int ori_w) {

    result.boxes.clear();
    result.keypoints.clear();

    const auto lb = compute_letterbox(ori_w, ori_h, cfg_.model_w, cfg_.model_h);

    auto scratch_guard = Util::make_rental_pool_guard(
            pose_scratch_pool_, "YoloUltralyticsPose: pose scratch pool checkout failed");
    scratch_guard.ptr->boxes.clear();
    scratch_guard.ptr->keypoints.clear();
    std::vector<BBox>& all_boxes = scratch_guard.ptr->boxes;
    std::vector<std::vector<Keypoint>>& all_kpts = scratch_guard.ptr->keypoints;

    for (size_t layer_id = 0; layer_id < NUM_LAYERS; ++layer_id) {
        const auto& layer = yolo_post_layers_[layer_id];
        float* conf_base = outputs.at(layer.conf_port);
        float* coord_base = outputs.at(layer.coord_port);
        float* kpt_base = outputs.at(layer.keypt_port);

        for (size_t i = 0; i < layer.height * layer.width; ++i) {
            float score = conf_base[i];

            // NOTE: use inv_conf_thres here because score is still raw (not applied sigmoid yet).
            // sigmoid is expensive and we only apply it if needed.
            if (score < smgr_->inv_conf_thres)
                continue;

            // convert score in [0,1] (e.g. apply sigmoid)
            score = smgr_->convert(score);

            // Get the specific anchor for this grid cell
            const Point2f& anchor = layer.anchors[i];

            // 1. Decode BBox (Distribution Focal Loss)
            std::array<float, 4> coord = MX::Prepost::Util::dfl(coord_base + i * COORD_FMAP_SIZE,
                                                                i / layer.width /* row */,
                                                                i % layer.width /* col */,
                                                                layer.stride,
                                                                cfg_.model_w,
                                                                cfg_.model_h);

            // Convert BBox to original image scale
            float x1 = (coord[0] - lb.pad_left) / lb.ratio;
            float y1 = (coord[1] - lb.pad_top) / lb.ratio;
            float x2 = (coord[2] - lb.pad_left) / lb.ratio;
            float y2 = (coord[3] - lb.pad_top) / lb.ratio;

            // Clip to image bounds
            x1 = std::max(0.0f, std::min(x1, (float)ori_w));
            y1 = std::max(0.0f, std::min(y1, (float)ori_h));
            x2 = std::max(0.0f, std::min(x2, (float)ori_w));
            y2 = std::max(0.0f, std::min(y2, (float)ori_h));

            // Drop invalid/degenerate boxes
            if (x2 <= x1 || y2 <= y1) {
                continue;
            }

            all_boxes.emplace_back(x1, y1, x2, y2, score, 0, "person");

            // 2. Decode Keypoints
            std::vector<Keypoint> kpts_per_box;
            for (size_t k = 0; k < NUM_KEYPOINTS; ++k) {
                // offset: Jump to grid cell 'i', then jump to keypoint 'k', each has (x, y, conf)
                int offset = (i * NUM_KEYPOINTS * 3) + (k * 3);

                float raw_x = kpt_base[offset];
                float raw_y = kpt_base[offset + 1];
                float kpt_conf_raw = kpt_base[offset + 2];

                // Only process if keypoint confidence is high enough
                if (kpt_conf_raw > smgr_->inv_conf_thres) {
                    // YOLOv8 Pose Decoding:
                    // 1. Multiply by 2.0 (Model output range is usually -0.5 to 1.5)
                    // 2. Add the grid center (Shift to absolute feature map position)
                    // 3. Multiply by stride (Scale up to model input size: e.g., 640x640)
                    //
                    // detail:
                    // https://docs.google.com/document/d/1ENBtyOH2IyDi_NlozJ2HvOvattHg9WLydIHEzjMoZY0/edit?usp=sharing
                    float kpt_x = (raw_x * 2.0f + (anchor.x - 0.5f)) * layer.stride;
                    float kpt_y = (raw_y * 2.0f + (anchor.y - 0.5f)) * layer.stride;

                    // 4. Recovery from Letterbox (Scale to original image pixels)
                    kpt_x = (kpt_x - lb.pad_left) / lb.ratio;
                    kpt_y = (kpt_y - lb.pad_top) / lb.ratio;

                    float kpt_conf = smgr_->convert(kpt_conf_raw);
                    kpts_per_box.emplace_back(kpt_x, kpt_y, kpt_conf);
                } else {
                    // Use a sentinel value for hidden/occluded keypoints
                    kpts_per_box.emplace_back(-1, -1, 0.0f);
                }
            }
            all_kpts.push_back(kpts_per_box);
        }
    }

    // apply NMS
    std::vector<int> keep_indices =
            MX::Prepost::Util::nms(all_boxes, cfg_.iou, cfg_.class_agnostic, cfg_.max_dets);

    // early exit
    int num_keep = static_cast<int>(keep_indices.size());
    if (num_keep == 0)
        return;

    // keep only the selected boxes
    result.boxes.reserve(keep_indices.size());
    for (int idx : keep_indices) {
        result.boxes.push_back(all_boxes[idx]);
    }

    // keep only the selected keypoints
    result.keypoints.reserve(keep_indices.size());
    for (int idx : keep_indices) {
        result.keypoints.push_back(all_kpts[idx]);
    }
}
