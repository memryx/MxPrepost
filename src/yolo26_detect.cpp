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

#include "yolo26_detect.h"

#include "config_finalizer.h"
#include "utils.h"



using namespace MX::Runtime;
using namespace MX::Prepost::Util;
using namespace MX::Prepost;

Yolo26Detect::Yolo26Detect(MX::Runtime::MxAcclBase* accl,
                                             const YoloUserConfig& user_cfg,
                                             const std::string& task) {

    // init settings from config
    cfg_ = ConfigFinalizer::finalize(accl, user_cfg);

    // init score manager
    smgr_ = std::make_unique<MX::Prepost::Util::ScoreManager>(cfg_.conf, cfg_.fast_sigmoid);

    //------------------------------------------------------------------------------------------------
    // derive shape config from:
    // * model input shape (from accl model_info)
    // * number of classes (from class_labels size)
    // * arch parameters: 
    //   * num layers
    //   * strides
    //   * DFL with X coord channels
    //   * total preds
    
    auto model_info = accl->get_model_info(cfg_.model_id);

    yolo_post_layers_.resize(NUM_LAYERS);

    std::vector<MX::Types::ShapeVector> expected_coord_shapes;
    std::vector<MX::Types::ShapeVector> expected_conf_shapes;

    // total_preds_ is the sum of (height * width) across all layers
    total_preds_ = 0;

    for(int i=0; i < NUM_LAYERS; ++i) {
        int stride = STRIDES[i];
        expected_coord_shapes.push_back({cfg_.model_h / stride, cfg_.model_w / stride, 1, COORD_FMAP_SIZE});
        expected_conf_shapes.push_back({cfg_.model_h / stride, cfg_.model_w / stride, 1, static_cast<int64_t>(cfg_.class_labels.size())});
        total_preds_ += (cfg_.model_h / stride) * (cfg_.model_w / stride);
    }

    // in case the user is forcing a particular mapping
    if(cfg_.override_layer_mapping.empty()){
        // match expected shapes with actual model input shapes to find the correct ports for each layer
        int num_ofmaps = model_info.out_featuremap_shapes.size();

        if(cfg_.class_labels.size() != COORD_FMAP_SIZE){
            for (int i=0; i < NUM_LAYERS; ++i){
                yolo_post_layers_[i].height = cfg_.model_h / STRIDES[i];
                yolo_post_layers_[i].width = cfg_.model_w / STRIDES[i];
                yolo_post_layers_[i].stride = STRIDES[i];

                bool found_coord = false;
                bool found_conf = false;
                for (int port = 0; port < num_ofmaps; ++port) {
                    const auto& actual_shape = model_info.out_featuremap_shapes[port];
                    if (actual_shape == expected_coord_shapes[i]) {
                        yolo_post_layers_[i].coord_port = port;
                        found_coord = true;
                    } else if (actual_shape == expected_conf_shapes[i]) {
                        yolo_post_layers_[i].conf_port = port;
                        found_conf = true;
                    }
                }

                if(!found_coord) {
                    throw std::runtime_error("Could not find coordinate output port for layer " + std::to_string(i) + ". Was looking for shape " + expected_coord_shapes[i].to_string());
                }
                if(!found_conf) {
                    std::string error_message = "Could not find confidence output port (for stride layer " + std::to_string(i) + "). Was looking for shape " + expected_conf_shapes[i].to_string() + ". \n\nThis may be because the number of class labels provided (" + std::to_string(cfg_.class_labels.size()) + ") does not match the model's expected number of classes for this layer. \n\nIf you are intentionally providing a different number of class labels (either as a list or a text file), please ensure that it matches the model's output shape.";
                    throw std::runtime_error(error_message);
                }
            }
        }
        else {
            // warning for the corner case
            std::cerr << "WARNING: Number of classes (" << cfg_.class_labels.size() << ") is equal to COORD_FMAP_SIZE (" << COORD_FMAP_SIZE << "). "
                      << "This may cause ambiguity in automatically identifying coordinate and confidence ports based on output shapes. "
                      << "Please provide an explicit mapping using the override_layer_mapping entry in YoloUserConfig."
                      << std::endl;

            // fallback to guesstimation based on port order (coord then conf)
            for (int i=0; i < NUM_LAYERS; ++i){
                yolo_post_layers_[i].height = cfg_.model_h / STRIDES[i];
                yolo_post_layers_[i].width = cfg_.model_w / STRIDES[i];
                yolo_post_layers_[i].stride = STRIDES[i];

                yolo_post_layers_[i].coord_port = i * 2; // even ports for coord
                yolo_post_layers_[i].conf_port = i * 2 + 1; // odd ports for conf
            }

            // double check that our guessed shapes match the expected shapes, if not, throw an error
            for (int i=0; i < NUM_LAYERS; ++i){
                const auto& coord_shape = model_info.out_featuremap_shapes[yolo_post_layers_[i].coord_port];
                const auto& conf_shape = model_info.out_featuremap_shapes[yolo_post_layers_[i].conf_port];

                if (coord_shape != expected_coord_shapes[i]) {
                    throw std::runtime_error("Guessed coordinate port shape does not match expected shape for layer " + std::to_string(i) + ". Guessed coord port: " + std::to_string(yolo_post_layers_[i].coord_port) + " with shape " + coord_shape.to_string() + " and expected shape " + expected_coord_shapes[i].to_string());
                }
                if (conf_shape != expected_conf_shapes[i]) {
                    throw std::runtime_error("Guessed confidence port shape does not match expected shape for layer " + std::to_string(i) + ". Guessed conf port: " + std::to_string(yolo_post_layers_[i].conf_port) + " with shape " + conf_shape.to_string() + " and expected shape " + expected_conf_shapes[i].to_string());
                }
            }
        }

    }
    else {
        // user provided mapping ( <stride (int), vect<coord name, conf name>> )
        // parse the model_info to find the ports corresponding to the provided layer_names,
        // then get assign those ports to the correct layer in yolo_post_layers_ based on the stride,
        // then double check that the shapes match the expected shapes for that stride/layer
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

            // assigns based on expected shapes
            yolo_post_layers_[layer_id].height = cfg_.model_h / stride;
            yolo_post_layers_[layer_id].width = cfg_.model_w / stride;
            yolo_post_layers_[layer_id].stride = stride;

            bool found_coord = false;
            bool found_conf = false;
            for (unsigned int port = 0; port < model_info.out_featuremap_shapes.size(); ++port) {
                const auto& actual_shape = model_info.out_featuremap_shapes[port];
                // find the output port by name AND matching shape
                if (model_info.output_layer_names[port] == layer_names[0]) {
                    if (actual_shape == expected_coord_shapes[layer_id]) {
                        yolo_post_layers_[layer_id].coord_port = port;
                        found_coord = true;
                    } else {
                        std::string error_message = "Output port " + std::to_string(port) + " with name " + layer_names[0] + " has shape "
                            + actual_shape.to_string() + " which does not match expected coordinate shape for stride " + std::to_string(stride) + " which is "
                            + expected_coord_shapes[layer_id].to_string();
                        throw std::runtime_error(error_message);
                    }
                } else if (model_info.output_layer_names[port] == layer_names[1]) {
                    if (actual_shape == expected_conf_shapes[layer_id]) {
                        yolo_post_layers_[layer_id].conf_port = port;
                        found_conf = true;
                    } else {
                        std::string error_message = "Output port " + std::to_string(port) + " with name " + layer_names[1] + " has shape "
                            + actual_shape.to_string() + " which does not match expected confidence shape for stride " + std::to_string(stride) + " which is "
                            + expected_conf_shapes[layer_id].to_string() + ". Did you specify the right number of classes? This model has " + std::to_string(cfg_.class_labels.size());
                        throw std::runtime_error(error_message);
                    }
                }
            }

            if(!found_coord) {
                throw std::runtime_error("Could not find coordinate output port for layer with stride " + std::to_string(stride) + ". No name match was found for the given " + layer_names[0]);
            }
            if(!found_conf) {
                throw std::runtime_error("Could not find confidence output port for layer with stride " + std::to_string(stride) + ". No name match was found for the given " + layer_names[1]);
            }

            num_found_layers++;
        }

        if(num_found_layers != NUM_LAYERS) {
            throw std::runtime_error("override_layer_mapping must contain entries for all " + std::to_string(NUM_LAYERS) + " layers. Found entries for " + std::to_string(num_found_layers) + " layers.");
        }

    }
//     //------------------------------------------------------------------------------------------------

    for (unsigned i = 0; i < rentalStoreSlots; ++i) {
        Util::BBoxScratch* slot = bbox_pool_.checkout_item();
        if (slot == nullptr) {
            throw std::runtime_error("Yolo26Detect: bbox pool warmup failed");
        }
        slot->boxes.reserve(total_preds_);
        bbox_pool_.return_item(slot);
    }

    const size_t label_cap = cfg_.valid_classes.size();
    for (unsigned i = 0; i < rentalStoreSlots; ++i) {
        std::vector<std::pair<int, float>>* slot = label_pool_.checkout_item();
        if (slot == nullptr) {
            throw std::runtime_error("Yolo26Detect: label rental store warmup failed");
        }
        slot->reserve(label_cap);
        label_pool_.return_item(slot);
    }
}

cv::Mat Yolo26Detect::preprocess(const cv::Mat& image) {
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

void Yolo26Detect::draw(cv::Mat& image, const Result& result) {
    for (const BBox& bbox : result.boxes) {
        MX::Prepost::Util::draw_bbox(image, bbox);
    }
}

void Yolo26Detect::postprocess(const std::vector<float*>&, Result&) {
    throw std::runtime_error(
            "postprocess(outputs, result) requires original image or (ori_w, ori_h). "
            "Use postprocess(outputs, result, original_image) or postprocess(outputs, result, ori_w, ori_h).");
}

void Yolo26Detect::postprocess(const std::vector<float*>& outputs,
                                        Result& result,
                                        const cv::Mat& original_image) {
    if (UNLIKELY(original_image.empty())) {
        throw std::invalid_argument("original_image must be non-empty for postprocess");
    }
    postprocess_impl(outputs, result, original_image.rows, original_image.cols);
}

void Yolo26Detect::postprocess(const std::vector<float*>& outputs,
                                        Result& result,
                                        int ori_h,
                                        int ori_w) {
    if (UNLIKELY(ori_w <= 0 || ori_h <= 0)) {
        throw std::invalid_argument("ori_w and ori_h must be > 0 for postprocess");
    }
    postprocess_impl(outputs, result, ori_h, ori_w);
}

void Yolo26Detect::postprocess_impl(const std::vector<float*>& outputs,
                                             Result& result,
                                             int ori_h,
                                             int ori_w) {

    result.boxes.clear();
    const auto lb = compute_letterbox(ori_w, ori_h, cfg_.model_w, cfg_.model_h);

    auto bbox_guard =
            Util::make_rental_pool_guard(bbox_pool_, "Yolo26Detect: bbox pool checkout failed");
    std::vector<BBox>& all_boxes = bbox_guard.ptr->boxes;
    all_boxes.clear();

    auto label_guard =
            Util::make_rental_pool_guard(label_pool_, "Yolo26Detect: label pool checkout failed");
    std::vector<std::pair<int, float>>& labels = *label_guard.ptr;

    for (size_t layer_id = 0; layer_id < NUM_LAYERS; ++layer_id) {
        const auto& layer = yolo_post_layers_[layer_id];
        float* conf_base = outputs.at(layer.conf_port);
        float* coord_base = outputs.at(layer.coord_port);

        for (size_t i = 0; i < layer.height * layer.width; ++i) {

            // Gather all labels above threshold (multi-label).
            // NOTE: use inv_conf_thres here because score is still raw (not applied sigmoid yet).
            // sigmoid is expensive and we only apply it if needed.
            float* class_scores = conf_base + i * cfg_.class_labels.size();
            MX::Prepost::Util::get_labels_by_mode(labels,
                                                  class_scores,
                                                  cfg_.valid_classes,
                                                  smgr_->inv_conf_thres,
                                                  cfg_.multi_label);

            if (labels.empty())
                continue;

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

            // Clip to image bounds
            coord[0] = std::max(0.0f, std::min(coord[0], (float)ori_w));
            coord[1] = std::max(0.0f, std::min(coord[1], (float)ori_h));
            coord[2] = std::max(0.0f, std::min(coord[2], (float)ori_w));
            coord[3] = std::max(0.0f, std::min(coord[3], (float)ori_h));

            // Drop invalid/degenerate boxes
            if (coord[2] <= coord[0] || coord[3] <= coord[1])
                continue;

            // Multi-label: one box per class score that passed threshold.
            for (const auto& [label, score] : labels) {
                all_boxes.emplace_back(coord[0],
                                       coord[1],
                                       coord[2],
                                       coord[3],
                                       smgr_->convert(score),
                                       label,
                                       cfg_.class_labels[label]);
            }
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

    // early exit
    int num_keep = static_cast<int>(keep_indices.size());
    if (num_keep == 0)
        return;

    result.boxes.reserve(keep_indices.size());
    for (int idx : keep_indices) {
        result.boxes.push_back(all_boxes[idx]);
    }
}
