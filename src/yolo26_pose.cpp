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

#include "yolo26_pose.h"

#include "config_finalizer.h"
#include "utils.h"



using namespace MX::Runtime;
using namespace MX::Prepost::Util;
using namespace MX::Prepost;

Yolo26Pose::Yolo26Pose(MX::Runtime::MxAcclBase* accl,
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
        MX::Prepost::Util::PoseScratch* slot = pose_scratch_pool_.checkout_item();
        if (slot == nullptr) {
            throw std::runtime_error("Yolo26Pose: pose rental store warmup failed");
        }
        slot->boxes.reserve(total_preds_);
        slot->keypoints.reserve(total_preds_);
        pose_scratch_pool_.return_item(slot);
    }
}
cv::Mat Yolo26Pose::preprocess(const cv::Mat& image) {
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

void Yolo26Pose::draw(cv::Mat& image, const Result& result) {
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
                           YOLO26_KEYPOINT_COLORS[i % YOLO26_KEYPOINT_COLORS.size()],
                           -1);
            }
        }
    }
}

void Yolo26Pose::postprocess(const std::vector<float*>&, Result&) {
    throw std::runtime_error(
            "postprocess(outputs, result) requires original image or (ori_w, ori_h). "
            "Use postprocess(outputs, result, original_image) or postprocess(outputs, result, ori_w, ori_h).");
}

void Yolo26Pose::postprocess(const std::vector<float*>& outputs,
                                        Result& result,
                                        const cv::Mat& original_image) {
    if (UNLIKELY(original_image.empty())) {
        throw std::invalid_argument("original_image must be non-empty for postprocess");
    }
    postprocess_impl(outputs, result, original_image.rows, original_image.cols);
}

void Yolo26Pose::postprocess(const std::vector<float*>& outputs,
                                        Result& result,
                                        int ori_h,
                                        int ori_w) {
    if (UNLIKELY(ori_w <= 0 || ori_h <= 0)) {
        throw std::invalid_argument("ori_w and ori_h must be > 0 for postprocess");
    }
    postprocess_impl(outputs, result, ori_h, ori_w);
}

void Yolo26Pose::postprocess_impl(const std::vector<float*>& outputs,
                                             Result& result,
                                             int ori_h,
                                             int ori_w) {

    result.boxes.clear();
    result.keypoints.clear();
    const auto lb = compute_letterbox(ori_w, ori_h, cfg_.model_w, cfg_.model_h);

    auto scratch_guard = MX::Prepost::Util::make_rental_pool_guard(
            pose_scratch_pool_, "Yolo26Pose: pose scratch pool checkout failed");
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
            if (score < smgr_->inv_conf_thres)
                continue;

            score = smgr_->convert(score);

            const float* p = coord_base + i * COORD_FMAP_SIZE;
            const float l = p[0];
            const float t = p[1];
            const float r = p[2];
            const float b = p[3];

            const float row = static_cast<float>(i / layer.width);
            const float col = static_cast<float>(i % layer.width);
            const float center_x = col + 0.5f;
            const float center_y = row + 0.5f;
            const float stride = static_cast<float>(layer.stride);

            std::array<float, 4> coord = {(center_x - l) * stride,
                                          (center_y - t) * stride,
                                          (center_x + r) * stride,
                                          (center_y + b) * stride};

            coord[0] = (coord[0] - lb.pad_left) / lb.ratio;
            coord[1] = (coord[1] - lb.pad_top) / lb.ratio;
            coord[2] = (coord[2] - lb.pad_left) / lb.ratio;
            coord[3] = (coord[3] - lb.pad_top) / lb.ratio;

            coord[0] = std::max(0.0f, std::min(coord[0], (float)ori_w - 1));
            coord[1] = std::max(0.0f, std::min(coord[1], (float)ori_h - 1));
            coord[2] = std::max(0.0f, std::min(coord[2], (float)ori_w - 1));
            coord[3] = std::max(0.0f, std::min(coord[3], (float)ori_h - 1));

            if (coord[2] <= coord[0] || coord[3] <= coord[1])
                continue;

            std::vector<Keypoint> kpts_per_box;
            kpts_per_box.reserve(NUM_KEYPOINTS);
            const Point2f& anchor = layer.anchors[i];
            for (size_t k = 0; k < NUM_KEYPOINTS; ++k) {
                int offset = (i * NUM_KEYPOINTS * 3) + (k * 3);

                float raw_x = kpt_base[offset];
                float raw_y = kpt_base[offset + 1];
                float kpt_conf_raw = kpt_base[offset + 2];

                if (kpt_conf_raw > smgr_->inv_conf_thres) {
                    // YOLO26 pose keypoints are decoded as grid-relative offsets without
                    // the YOLOv8/11 extra "*2" scaling term.
                    float kpt_x = (raw_x + (anchor.x)) * layer.stride;
                    float kpt_y = (raw_y + (anchor.y)) * layer.stride;

                    kpt_x = (kpt_x - lb.pad_left) / lb.ratio;
                    kpt_y = (kpt_y - lb.pad_top) / lb.ratio;

                    float kpt_conf = smgr_->convert(kpt_conf_raw);
                    kpts_per_box.emplace_back(kpt_x, kpt_y, kpt_conf);
                } else {
                    kpts_per_box.emplace_back(-1, -1, 0.0f);
                }
            }

            all_boxes.emplace_back(coord[0],
                                   coord[1],
                                   coord[2],
                                   coord[3],
                                   score,
                                   0,
                                   "person");
            all_kpts.push_back(std::move(kpts_per_box));
        }
    }

    // YOLO26 path: no NMS. Sort by confidence and keep top-k.
    std::vector<int> keep_indices(all_boxes.size());
    std::iota(keep_indices.begin(), keep_indices.end(), 0);
    std::sort(keep_indices.begin(), keep_indices.end(), [&](int a, int b) {
        return all_boxes[static_cast<size_t>(a)].conf > all_boxes[static_cast<size_t>(b)].conf;
    });
    if (keep_indices.size() > static_cast<size_t>(cfg_.max_dets)) {
        keep_indices.resize(static_cast<size_t>(cfg_.max_dets));
    }

    if (keep_indices.empty())
        return;

    result.boxes.reserve(keep_indices.size());
    result.keypoints.reserve(keep_indices.size());
    for (int idx : keep_indices) {
        result.boxes.push_back(all_boxes[static_cast<size_t>(idx)]);
        result.keypoints.push_back(std::move(all_kpts[static_cast<size_t>(idx)]));
    }
}
