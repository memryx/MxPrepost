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

#include "config_finalizer.h"

#include <memx/accl/MxAcclBase.h>

#include <fstream>
using namespace MX::Runtime;
using namespace MX::Prepost;

namespace {  // anonymous namespace for helper functions
    std::vector<std::string> _load_classes(const std::string& file_path) {
        std::vector<std::string> class_labels;
        std::ifstream file(file_path);

        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << file_path << std::endl;
            return class_labels;
        }

        std::string line;
        while (std::getline(file, line)) {
            // Optional: Skip empty lines
            if (!line.empty()) {
                class_labels.push_back(line);
            }
        }

        file.close();
        return class_labels;
    }
}

YoloFinalConfig ConfigFinalizer::finalize(MX::Runtime::MxAcclBase* accl, const YoloUserConfig& user) {
    YoloFinalConfig finalcfg;

    MX::Types::MxModelInfo model_info = accl->get_model_info(user.model_id);
    if (model_info.use_model_shape_in == true) {
        throw std::runtime_error(
                "use_model_shape of input must be false for Yolo models in MxPrepost");
    }

    if (model_info.use_model_shape_out == true) {
        throw std::runtime_error(
                "use_model_shape of output must be false for Yolo models in MxPrepost");
    }
    
    if (user.max_dets <= 0) {
        throw std::invalid_argument("max_dets must be > 0.");
    }
    
    finalcfg.conf = user.conf;
    finalcfg.iou = user.iou;
    finalcfg.max_dets = user.max_dets;
    finalcfg.class_agnostic = user.class_agnostic;
    finalcfg.fast_sigmoid = user.fast_sigmoid;
    finalcfg.multi_label = user.multi_label;
    finalcfg.model_id = user.model_id;

    // For Class Labels
    if (user.classmap_path.empty() && user.custom_class_labels.empty()) {
        for (const auto& label : COCO_NAMES) {
            finalcfg.class_labels.push_back(label);
        }
    } else if (user.classmap_path.empty() == false && user.custom_class_labels.empty()) {
        // read from file
        finalcfg.class_labels = _load_classes(user.classmap_path);
    } else if (user.custom_class_labels.empty() == false && user.classmap_path.empty()) {
        // read from user input
        finalcfg.class_labels = user.custom_class_labels;
    } else {
        throw std::invalid_argument(
                "Both classmap_path and custom_class_labels are set. Please provide only one source of class labels.");
    }

    // For Valid Classes
    if (user.valid_classes.empty()) {
        for (int i = 0; i < (int) finalcfg.class_labels.size(); ++i) {
            finalcfg.valid_classes.push_back(i);
        }
    } else {
        for (int cls : user.valid_classes) {
            if (cls < 0 || cls >= (int) finalcfg.class_labels.size()) {
                throw std::invalid_argument("valid_classes contains invalid class ID: " +
                                            std::to_string(cls));
            }
            finalcfg.valid_classes.push_back(cls);
        }
    }

    // Get model input dimensions
    finalcfg.model_h = model_info.in_featuremap_shapes[0][0];
    finalcfg.model_w = model_info.in_featuremap_shapes[0][1];

    // Copy override_layer_mapping
    finalcfg.override_layer_mapping = user.override_layer_mapping;

    return finalcfg;
}
