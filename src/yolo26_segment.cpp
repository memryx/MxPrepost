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

#include "yolo26_segment.h"

#include "config_finalizer.h"
#include "utils.h"

#include <algorithm>



using namespace MX::Runtime;
using namespace MX::Prepost::Util;
using namespace MX::Prepost;

Yolo26Segment::Yolo26Segment(MX::Runtime::MxAcclBase* accl,
                                             const YoloUserConfig& user_cfg,
                                             const std::string& task) {

    // init settings from config
    cfg_ = ConfigFinalizer::finalize(accl, user_cfg);

    // init score manager
    smgr_ = std::make_unique<MX::Prepost::Util::ScoreManager>(cfg_.conf, cfg_.fast_sigmoid);

    // autocalculate the layer shapes and corresponding ports based on model output shapes
    // and expected shapes, unless user provided an explicit mapping in the config (override_layer_mapping)
    auto model_info = accl->get_model_info(cfg_.model_id);

    yolo_post_layers_.resize(NUM_LAYERS);

    std::vector<MX::Types::ShapeVector> expected_coord_shapes;
    std::vector<MX::Types::ShapeVector> expected_conf_shapes;
    std::vector<MX::Types::ShapeVector> expected_mask_coef_shapes;
    MX::Types::ShapeVector              expected_mask_proto_shape(cfg_.model_h / MASK_DIV_FACTOR, cfg_.model_w / MASK_DIV_FACTOR, 1, MASK_CHANNELS);

    mask_proto_port_ = 0;
    mask_proto_h_ = 0;
    mask_proto_w_ = 0;


    // expected coord/conf are the same
    // mask channels are the same across layers, but height and width depend on the stride
    for(int i=0; i < NUM_LAYERS; ++i) {
        int stride = STRIDES[i];
        expected_coord_shapes.push_back({cfg_.model_h / stride, cfg_.model_w / stride, 1, COORD_FMAP_SIZE});
        expected_conf_shapes.push_back({cfg_.model_h / stride, cfg_.model_w / stride, 1, static_cast<int64_t>(cfg_.class_labels.size())});
        expected_mask_coef_shapes.push_back({cfg_.model_h / stride, cfg_.model_w / stride, 1, MASK_CHANNELS});
        total_preds_ += (cfg_.model_h / stride) * (cfg_.model_w / stride);
    }

    // in case the user is forcing a particular mapping
    if(cfg_.override_layer_mapping.empty()){
        
        // match expected shapes with actual model input shapes to find the correct ports for each layer
        int num_ofmaps = model_info.out_featuremap_shapes.size();

        if(cfg_.class_labels.size() != COORD_FMAP_SIZE && cfg_.class_labels.size() != MASK_CHANNELS){
            // if the number of classes is different from the number of coord channels and mask coef channels, then we can unambiguously identify the coord, conf, and mask coef ports based on their shapes
            for (int i=0; i < NUM_LAYERS; ++i){
                int stride = STRIDES[i];
                yolo_post_layers_[i].height = cfg_.model_h / stride;
                yolo_post_layers_[i].width = cfg_.model_w / stride;
                yolo_post_layers_[i].stride = stride;

                bool found_coord = false;
                bool found_conf = false;
                bool found_mask_coef = false;
                bool found_mask_proto = false;
                for (int port = 0; port < num_ofmaps; ++port) {
                    const auto& actual_shape = model_info.out_featuremap_shapes[port];
                    if (actual_shape == expected_coord_shapes[i]) {
                        yolo_post_layers_[i].coord_port = port;
                        found_coord = true;
                    } else if (actual_shape == expected_conf_shapes[i]) {
                        yolo_post_layers_[i].conf_port = port;
                        found_conf = true;
                    } else if (actual_shape == expected_mask_coef_shapes[i]) {
                        yolo_post_layers_[i].mask_coef_port = port;
                        found_mask_coef = true;
                    } else if (actual_shape == expected_mask_proto_shape) {
                        yolo_post_layers_[i].mask_proto_port = port;
                        found_mask_proto = true;
                        
                        // psa: this will get called multiple times
                        mask_proto_port_ = port;
                        mask_proto_h_ = expected_mask_proto_shape[0];
                        mask_proto_w_ = expected_mask_proto_shape[1];
                    }
                }

                if(!found_coord) {
                    throw std::runtime_error("Could not find coordinate output port for layer " + std::to_string(i) + ". Was looking for shape " + expected_coord_shapes[i].to_string());
                }
                if(!found_conf) {
                    std::string error_message = "Could not find confidence output port (for stride layer " + std::to_string(i) + "). Was looking for shape " + expected_conf_shapes[i].to_string() + ". \n\nThis may be because the number of class labels provided (" + std::to_string(cfg_.class_labels.size()) + ") does not match the model's expected number of classes for this layer. \n\nIf you are intentionally providing a different number of class labels (either as a list or a text file), please ensure that it matches the model's output shape.";
                    throw std::runtime_error(error_message);
                }
                if(!found_mask_coef) {
                    throw std::runtime_error("Could not find mask coefficient output port for layer " + std::to_string(i) + ". Was looking for shape " + expected_mask_coef_shapes[i].to_string());
                }
                if(!found_mask_proto) {
                    throw std::runtime_error("Could not find mask proto output port for layer " + std::to_string(i) + ". Was looking for shape " + expected_mask_proto_shape.to_string());
                }
            }

        }
        else {
            // else we print a warning and we guess based on the expected order of: coord, conf, mask_coef
            // mask_proto will never match the expected shape of the other layers so that can be identified unambiguously
            std::cerr << "Warning: Number of classes matches the number of coord channels and mask coef channels, so unable to unambiguously identify output ports based on shape. Will attempt to guess the ports based on the expected order of coord, conf, mask_coef in the model outputs. If this is incorrect, please provide an explicit mapping using the override_layer_mapping entry in YoloUserConfig." << std::endl;
            for (int i=0; i < NUM_LAYERS; ++i){
                int stride = STRIDES[i];
                yolo_post_layers_[i].height = cfg_.model_h / stride;
                yolo_post_layers_[i].width = cfg_.model_w / stride;
                yolo_post_layers_[i].stride = stride;

                yolo_post_layers_[i].coord_port = i * 3; // every 3 ports is a new layer, and coord is first
                yolo_post_layers_[i].conf_port = i * 3 + 1; // conf is second
                yolo_post_layers_[i].mask_coef_port = i * 3 + 2; // mask coef is third

                // psa: this will get called multiple times
                for (int port = 0; port < num_ofmaps; ++port) {
                    const auto& actual_shape = model_info.out_featuremap_shapes[port];
                    if (actual_shape == expected_mask_proto_shape) {
                        yolo_post_layers_[i].mask_proto_port = port;

                        mask_proto_port_ = port;
                        mask_proto_h_ = expected_mask_proto_shape[0];
                        mask_proto_w_ = expected_mask_proto_shape[1];
                        break;
                    }
                }

            }

            // verify each yolo_post_layers_[] has the expected shapes for coord, conf, mask_coef, and mask_proto
            for (int i=0; i < NUM_LAYERS; ++i){

                // check for any ports == -1 (meaning they were not found)
                if (yolo_post_layers_[i].coord_port == -1) {
                    throw std::runtime_error("Guessed coordinate port for layer " + std::to_string(i) + " was not found. Please check your model outputs and consider providing an explicit mapping using the override_layer_mapping entry in YoloUserConfig.");
                }
                if (yolo_post_layers_[i].conf_port == -1) {
                    throw std::runtime_error("Guessed confidence port for layer " + std::to_string(i) + " was not found. Please check your model outputs and consider providing an explicit mapping using the override_layer_mapping entry in YoloUserConfig.");
                }
                if (yolo_post_layers_[i].mask_coef_port == -1) {
                    throw std::runtime_error("Guessed mask coefficient port for layer " + std::to_string(i) + " was not found. Please check your model outputs and consider providing an explicit mapping using the override_layer_mapping entry in YoloUserConfig.");
                }
                if (yolo_post_layers_[i].mask_proto_port == -1) {
                    throw std::runtime_error("Guessed mask proto port for layer " + std::to_string(i) + " was not found. Please check your model outputs and consider providing an explicit mapping using the override_layer_mapping entry in YoloUserConfig.");
                }

                // will naturally crash the program if the layers weren't found earlier....
                const auto& coord_shape = model_info.out_featuremap_shapes[yolo_post_layers_[i].coord_port];
                const auto& conf_shape = model_info.out_featuremap_shapes[yolo_post_layers_[i].conf_port];
                const auto& mask_coef_shape = model_info.out_featuremap_shapes[yolo_post_layers_[i].mask_coef_port];
                const auto& mask_proto_shape = model_info.out_featuremap_shapes[yolo_post_layers_[i].mask_proto_port];

                if(coord_shape != expected_coord_shapes[i]) {
                    throw std::runtime_error("Guessed coordinate port shape does not match expected shape for layer " + std::to_string(i) + ". Guessed coord port: " + std::to_string(yolo_post_layers_[i].coord_port) + " with shape " + coord_shape.to_string() + " and expected shape " + expected_coord_shapes[i].to_string());
                }

                if(conf_shape != expected_conf_shapes[i]) {
                    throw std::runtime_error("Guessed confidence port shape does not match expected shape for layer " + std::to_string(i) + ". Guessed conf port: " + std::to_string(yolo_post_layers_[i].conf_port) + " with shape " + conf_shape.to_string() + " and expected shape " + expected_conf_shapes[i].to_string());
                }

                if(mask_coef_shape != expected_mask_coef_shapes[i]) {
                    throw std::runtime_error("Guessed mask coefficient port shape does not match expected shape for layer " + std::to_string(i) + ". Guessed mask coef port: " + std::to_string(yolo_post_layers_[i].mask_coef_port) + " with shape " + mask_coef_shape.to_string() + " and expected shape " + expected_mask_coef_shapes[i].to_string());
                }

                if(mask_proto_shape != expected_mask_proto_shape) {
                    throw std::runtime_error("Guessed mask proto port shape does not match expected shape for layer " + std::to_string(i) + ". Guessed mask proto port: " + std::to_string(yolo_post_layers_[i].mask_proto_port) + " with shape " + mask_proto_shape.to_string() + " and expected shape " + expected_mask_proto_shape.to_string());
                }
            }
                
        } // override_layer_mapping

    }
    else {
        // user provided mapping ( <stride (int), vect<coord name, conf name, mask coef name>, and special case stride=0 for global mask_proto)
        // parse the model_info to find the ports corresponding to the provided layer_names,
        // then get assign those ports to the correct layer in yolo_post_layers_ based on the stride (or 0 for mask_proto),
        // then double check that the shapes match the expected shapes for that stride/layer
        int num_found_layers = 0;

        for (const auto& [stride, layer_names] : cfg_.override_layer_mapping) {
            int layer_id = -1;
            if (stride == 0) {
                // special case for global mask_proto
                layer_id = 0; // assign to layer 0 but will be used globally
            } else {
                for (int i=0; i < NUM_LAYERS; ++i) {
                    if (STRIDES[i] == stride) {
                        layer_id = i;
                        break;
                    }
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
                error_message += ", or 0 for global mask_proto";
                throw std::runtime_error(error_message);
            }

            bool found_coord = false;
            bool found_conf = false;
            bool found_mask_coef = false;
            bool found_mask_proto = false;
            if (stride == 0) {
                // special case for global mask_proto where we only check for the mask_proto shape and port, not coord/conf ports
                for (unsigned int port = 0; port < model_info.out_featuremap_shapes.size(); ++port) {
                    if (model_info.output_layer_names[port] == layer_names[0]) { // vect should only contain one name which is for the mask_proto
                        const auto& actual_shape = model_info.out_featuremap_shapes[port];
                        if (actual_shape == expected_mask_proto_shape) {
                            // assign mask_proto_port in all yolo_post_layers_
                            for (int i = 0; i < NUM_LAYERS; ++i) {
                                yolo_post_layers_[i].mask_proto_port = port;
                            }
                            mask_proto_port_ = port;
                            mask_proto_h_ = expected_mask_proto_shape[0];
                            mask_proto_w_ = expected_mask_proto_shape[1];
                            found_mask_proto = true;
                            break;
                        } else {
                            std::string error_message = "Output port " + std::to_string(port) + " has shape " + actual_shape.to_string() 
                                + " which does not match expected mask proto shape " + expected_mask_proto_shape.to_string() 
                                + " for the global mask_proto. Please check your override_layer_mapping entry in YoloUserConfig.";
                            throw std::runtime_error(error_message);
                        }
                    }
                }

                if(!found_mask_proto) {
                    throw std::runtime_error("Could not find output port for global mask proto. Please check your override_layer_mapping entry in YoloUserConfig.");
                }

            }
            else { // coord, conf, mask_coef
                for (unsigned int port = 0; port < model_info.out_featuremap_shapes.size(); ++port) {
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
                                + expected_conf_shapes[layer_id].to_string() + ". Did you specify the right number of classes? This model has " + std::to_string(cfg_.class_labels.size());
                            throw std::runtime_error(error_message);
                        }
                    } else if (model_info.output_layer_names[port] == layer_names[2]) { // mask_coef
                        if (actual_shape == expected_mask_coef_shapes[layer_id]) {
                            yolo_post_layers_[layer_id].mask_coef_port = port;
                            found_mask_coef = true;
                        } else {
                            std::string error_message = "Output port " + std::to_string(port) + " with name " + layer_names[2] + " has shape "
                                + actual_shape.to_string() + " which does not match expected mask coefficient shape for stride " + std::to_string(stride) + " which is "
                                + expected_mask_coef_shapes[layer_id].to_string();
                            throw std::runtime_error(error_message);
                        }
                    }
                }
            }

            // if stride=0, check found_mask_proto, else check found_coord, found_conf, and found_mask_coef
            if (stride == 0) {
                if(!found_mask_proto) {
                    throw std::runtime_error("Could not find output port for global mask proto. Please check your override_layer_mapping entry in YoloUserConfig.");
                }
            }
            else {
                if(!found_coord) {
                    throw std::runtime_error("Could not find coordinate output port for layer with stride " + std::to_string(stride) + ". No name match was found for the given " + layer_names[0]);
                }
                if(!found_conf) {
                    throw std::runtime_error("Could not find confidence output port for layer with stride " + std::to_string(stride) + ". No name match was found for the given " + layer_names[1]);
                }
                if(!found_mask_coef) {
                    throw std::runtime_error("Could not find mask coefficient output port for layer with stride " + std::to_string(stride) + ". No name match was found for the given " + layer_names[2]);
                }
            }

        } // for each entry in override_layer_mapping
        

        // make sure we matched all needed layers (STRIDES * 3) + 1
        if(num_found_layers != (NUM_LAYERS * 3) + 1) {
            std::string error_message = "override_layer_mapping is missing some layers. Expected " + std::to_string(NUM_LAYERS * 3 + 1) + " layers (coord, conf, and mask_coef for each stride layer, plus global mask_proto), but found " + std::to_string(num_found_layers) + ". Please check your override_layer_mapping entry in YoloUserConfig.";
            throw std::runtime_error(error_message);
        }

    } // override_layer_mapping

    inv_mask_thresh_ = smgr_->invert(MASK_THRESH);

    for (unsigned i = 0; i < rentalStoreSlots; ++i) {
        Util::SegmentScratch* slot = segment_scratch_pool_.checkout_item();
        if (slot == nullptr) {
            throw std::runtime_error("Yolo26Segment: segment rental store warmup failed");
        }
        slot->boxes.reserve(total_preds_);
        slot->mask_coefs.reserve(total_preds_);
        segment_scratch_pool_.return_item(slot);
    }

    const size_t label_cap = cfg_.valid_classes.size();
    for (unsigned i = 0; i < rentalStoreSlots; ++i) {
        std::vector<std::pair<int, float>>* slot = label_pool_.checkout_item();
        if (slot == nullptr) {
            throw std::runtime_error("Yolo26Segment: label rental store warmup failed");
        }
        slot->reserve(label_cap);
        label_pool_.return_item(slot);
    }
}

cv::Mat Yolo26Segment::preprocess(const cv::Mat& image) {
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

void Yolo26Segment::draw(cv::Mat& image, const Result& result) {
    for (const Mask& mask : result.masks) {
        MX::Prepost::Util::draw_mask(image, mask);
    }
}

void Yolo26Segment::postprocess(const std::vector<float*>&, Result&) {
    throw std::runtime_error(
            "postprocess(outputs, result) requires original image or (ori_w, ori_h). "
            "Use postprocess(outputs, result, original_image) or postprocess(outputs, result, ori_w, ori_h).");
}

void Yolo26Segment::postprocess(const std::vector<float*>& outputs,
                                        Result& result,
                                        const cv::Mat& original_image) {
    if (UNLIKELY(original_image.empty())) {
        throw std::invalid_argument("original_image must be non-empty for postprocess");
    }
    postprocess_impl(outputs, result, original_image.rows, original_image.cols);
}

void Yolo26Segment::postprocess(const std::vector<float*>& outputs,
                                        Result& result,
                                        int ori_h,
                                        int ori_w) {
    if (UNLIKELY(ori_w <= 0 || ori_h <= 0)) {
        throw std::invalid_argument("ori_w and ori_h must be > 0 for postprocess");
    }
    postprocess_impl(outputs, result, ori_h, ori_w);
}

void Yolo26Segment::postprocess_impl(const std::vector<float*>& outputs,
                                             Result& result,
                                             int ori_h,
                                             int ori_w) {

    result.boxes.clear();
    result.masks.clear();

    const auto lb = compute_letterbox(ori_w, ori_h, cfg_.model_w, cfg_.model_h);

    auto scratch_guard = Util::make_rental_pool_guard(
            segment_scratch_pool_, "Yolo26Segment: segment scratch pool checkout failed");
    scratch_guard.ptr->boxes.clear();
    scratch_guard.ptr->mask_coefs.clear();
    std::vector<BBox>& all_boxes = scratch_guard.ptr->boxes;
    std::vector<float*>& all_mask_coefs = scratch_guard.ptr->mask_coefs;

    auto label_guard = Util::make_rental_pool_guard(
            label_pool_, "Yolo26Segment: label pool checkout failed");
    std::vector<std::pair<int, float>>& labels = *label_guard.ptr;

    for (size_t layer_id = 0; layer_id < NUM_LAYERS; ++layer_id) {
        const auto& layer = yolo_post_layers_[layer_id];
        float* conf_base = outputs.at(layer.conf_port);
        float* coord_base = outputs.at(layer.coord_port);
        float* mask_coef_base = outputs.at(layer.mask_coef_port);

        for (size_t i = 0; i < layer.height * layer.width; ++i) {

            // NOTE: use inv_conf_thres because score is still raw (not applied sigmoid yet).
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

            coord[0] = std::max(0.0f, std::min(coord[0], (float)ori_w));
            coord[1] = std::max(0.0f, std::min(coord[1], (float)ori_h));
            coord[2] = std::max(0.0f, std::min(coord[2], (float)ori_w));
            coord[3] = std::max(0.0f, std::min(coord[3], (float)ori_h));

            if (coord[2] <= coord[0] || coord[3] <= coord[1])
                continue;

            for (const auto& [label, score] : labels) {
                all_boxes.emplace_back(coord[0],
                                       coord[1],
                                       coord[2],
                                       coord[3],
                                       smgr_->convert(score),
                                       label,
                                       cfg_.class_labels[label]);
                all_mask_coefs.emplace_back(mask_coef_base + i * MASK_CHANNELS);
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

    const int num_keep = static_cast<int>(keep_indices.size());
    if (num_keep == 0)
        return;

    result.boxes.reserve(static_cast<size_t>(num_keep));
    std::vector<float*> kept_mask_coefs;
    kept_mask_coefs.reserve(static_cast<size_t>(num_keep));
    for (int idx : keep_indices) {
        result.boxes.push_back(all_boxes[static_cast<size_t>(idx)]);
        kept_mask_coefs.push_back(all_mask_coefs[static_cast<size_t>(idx)]);
    }

    cv::Mat mask_coefs_mat(MASK_CHANNELS, num_keep, CV_32F);
    for (int i = 0; i < num_keep; ++i) {
        cv::Mat col_header = mask_coefs_mat.col(i);
        cv::Mat src_col(MASK_CHANNELS, 1, CV_32F, kept_mask_coefs[static_cast<size_t>(i)]);
        src_col.copyTo(col_header);
    }

    cv::Mat mask_proto_raw(mask_proto_w_ * mask_proto_h_,
                           MASK_CHANNELS,
                           CV_32F,
                           (void*)outputs[mask_proto_port_]);
    cv::Mat mask_proto = mask_proto_raw.clone();
    mask_proto.forEach<float>([this](float& v, const int*) {
        v = v * smgr_->convert(v);
    });
    cv::Mat raw_masks = mask_proto * mask_coefs_mat;

    int pad_left = lb.pad_left;
    int pad_top = lb.pad_top;
    int unpad_w = (int)std::round(ori_w * lb.ratio);
    int unpad_h = (int)std::round(ori_h * lb.ratio);

    pad_left = std::clamp(pad_left, 0, cfg_.model_w);
    pad_top = std::clamp(pad_top, 0, cfg_.model_h);
    unpad_w = std::clamp(unpad_w, 0, cfg_.model_w - pad_left);
    unpad_h = std::clamp(unpad_h, 0, cfg_.model_h - pad_top);

    if (unpad_w <= 0 || unpad_h <= 0)
        return;

    cv::Rect unpad_roi(pad_left, pad_top, unpad_w, unpad_h);

    for (int i = 0; i < num_keep; ++i) {
        const BBox& box = result.boxes[i];

        cv::Mat mask_proto_res =
                raw_masks.col(i).clone().reshape(1, mask_proto_h_);

        cv::Mat mask_model;
        cv::resize(mask_proto_res,
                   mask_model,
                   cv::Size(cfg_.model_w, cfg_.model_h),
                   0,
                   0,
                   cv::INTER_LINEAR);

        cv::Mat mask_unpad = mask_model(unpad_roi);

        cv::Mat mask_ori;
        cv::resize(mask_unpad,
                   mask_ori,
                   cv::Size(ori_w, ori_h),
                   0,
                   0,
                   cv::INTER_LINEAR);

        int x1 = std::clamp((int)std::round(box.x_min), 0, ori_w);
        int y1 = std::clamp((int)std::round(box.y_min), 0, ori_h);
        int x2 = std::clamp((int)std::round(box.x_max), 0, ori_w);
        int y2 = std::clamp((int)std::round(box.y_max), 0, ori_h);

        if (x2 <= x1 || y2 <= y1)
            continue;

        cv::Rect bbox_roi(x1, y1, x2 - x1, y2 - y1);
        cv::Mat mask_roi = mask_ori(bbox_roi);
        cv::Mat binary_mask;
        cv::threshold(mask_roi, binary_mask, MASK_THRESH, 255, cv::THRESH_BINARY);
        binary_mask.convertTo(binary_mask, CV_8U);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binary_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        int largest_contour_idx = MX::Prepost::Util::get_largest_contour_idx(contours);

        if (largest_contour_idx < 0)
            continue;

        const auto& contour = contours[static_cast<size_t>(largest_contour_idx)];
        Mask mask_struct;
        mask_struct.cls_id = box.cls_id;
        mask_struct.cls_name = box.cls_name;
        mask_struct.xys.reserve(contour.size());

        for (const auto& p : contour) {
            mask_struct.xys.push_back({p.x + x1, p.y + y1});
        }

        if (!mask_struct.xys.empty()) {
            result.masks.push_back(mask_struct);
        }
    }
}
