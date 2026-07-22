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
#include <array>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <unordered_set>
#include <map>
#include <vector>
#include <utility>

namespace MX {
    namespace Prepost {

        /** RentalStore pool depth (max concurrent postprocess scratch checkouts per buffer type). */
        constexpr unsigned rentalStoreSlots = 16;

        // Labels of COCO dataset, COCO 2014 and 2017 uses the same images but
        // different train/val/test splits. Also, COCO defines 91 classes but the data
        // only uses 80 classes.
        constexpr int COCO_CLASS_NUMBER = 80;
        inline const char* COCO_NAMES[COCO_CLASS_NUMBER]{
                "person",        "bicycle",       "car",           "motorbike",
                "aeroplane",     "bus",           "train",         "truck",
                "boat",          "traffic light", "fire hydrant",  "stop sign",
                "parking meter", "bench",         "bird",          "cat",
                "dog",           "horse",         "sheep",         "cow",
                "elephant",      "bear",          "zebra",         "giraffe",
                "backpack",      "umbrella",      "handbag",       "tie",
                "suitcase",      "frisbee",       "skis",          "snowboard",
                "sports ball",   "kite",          "baseball bat",  "baseball glove",
                "skateboard",    "surfboard",     "tennis racket", "bottle",
                "wine glass",    "cup",           "fork",          "knife",
                "spoon",         "bowl",          "banana",        "apple",
                "sandwich",      "orange",        "broccoli",      "carrot",
                "hot dog",       "pizza",         "donut",         "cake",
                "chair",         "sofa",          "pottedplant",   "bed",
                "diningtable",   "toilet",        "tvmonitor",     "laptop",
                "mouse",         "remote",        "keyboard",      "cell phone",
                "microwave",     "oven",          "toaster",       "sink",
                "refrigerator",  "book",          "clock",         "vase",
                "scissors",      "teddy bear",    "hair drier",    "toothbrush",
        };


        /**
         * @brief Struct representing a bounding box detection result, with multiple representations (xyxy and xywh) for convenience.
         */
        struct BBox {

            float x_min; ///< Minimum x-coordinate (left) of the bounding box
            float y_min; ///< Minimum y-coordinate (top) of the bounding box
            float x_max; ///< Maximum x-coordinate (right) of the bounding box
            float y_max; ///< Maximum y-coordinate (bottom) of the bounding box

            // Bounding box representations
            std::array<float, 4> xyxy;  ///< (x_min, y_min, x_max, y_max)
            std::array<float, 4> xywh;  ///< (x_center, y_center, width, height)

            float conf = 0.0f;     ///< confidence score
            int cls_id = -1;       ///< class index
            std::string cls_name;  ///< class name as string

            // Default constructor
            BBox() = default;

            // Constructor with xyxy
            BBox(float x_min_,
                 float y_min_,
                 float x_max_,
                 float y_max_,
                 float conf_,
                 int cls_id_,
                 const std::string& cls_name_ = "") :
                x_min(x_min_), y_min(y_min_), x_max(x_max_), y_max(y_max_), conf(conf_),
                cls_id(cls_id_), cls_name(cls_name_) {
                update_xyxy();
                update_xywh();
            }

          private:
            void update_xyxy() {
                xyxy[0] = x_min;
                xyxy[1] = y_min;
                xyxy[2] = x_max;
                xyxy[3] = y_max;
            }

            void update_xywh() {
                xywh[0] = (x_min + x_max) * 0.5f;  // x_center
                xywh[1] = (y_min + y_max) * 0.5f;  // y_center
                xywh[2] = x_max - x_min;           // width
                xywh[3] = y_max - y_min;           // height
            }
        };

        struct Point {
            int x;
            int y;
        };

        /**
         * @brief Struct representing a point with floating-point coordinates, used for mask polygon and pose keypoints.
         */
        struct Point2f {
            float x; ///< X coordinate of the point
            float y; ///< Y coordinate of the point
        };

        /**
         * @brief Struct representing a detected segmentation mask, defined by a polygon of (x, y) coordinates and associated class information.
         */
        struct Mask {
            std::vector<Point2f> xys;  ///< List of (x, y) coordinates representing the polygon of the detected mask. Each coordinate is a Point2f object.
            int cls_id; ///< Class ID of the detected mask
            std::string cls_name; ///< Class name as string of the detected mask
        };

        /**
         * @brief Struct representing a detected keypoint for pose estimation, defined by its (x, y) coordinates and confidence score.
         */
        struct Keypoint {
            Point2f xy; ///< (x, y) coordinates of the pose keypoint
            float conf; ///< Confidence score of the detected pose keypoint

            Keypoint(Point2f xy, float conf) : xy{xy}, conf(conf) {
            }
            Keypoint(float x, float y, float conf) : xy{Point2f{x, y}}, conf(conf) {
            }
        };

        /**
         * @brief Struct representing the post-processing result for a single input image, containing detected bounding boxes, masks, and/or keypoints.
         */
        struct Result {
            std::vector<BBox> boxes; ///< List of detected bounding boxes in the input image, each represented as a BBox struct.
            std::vector<Mask> masks; ///< List of detected segmentation masks in the input image, each represented as a Mask struct.
            std::vector<std::vector<Keypoint>> keypoints; ///< List of detected keypoints for pose estimation in the input image. Each element in the outer vector corresponds to a detected instance (e.g., a person), and contains a vector of Keypoint structs representing the keypoints for that instance.
        };




        /**
         * @brief Struct representing the user-provided configuration for YOLO post-processing, including thresholds, class labels, and model/layer mapping overrides.
         */
        struct YoloUserConfig {
            float conf = 0.3f;  ///< Confidence threshold for filtering detections. Default 0.3
            float iou = 0.4f;   ///< IoU threshold for NMS. Default 0.4
            int max_dets = 300; ///< Maximum number of detections to keep after NMS. Default 300.

            std::string classmap_path = ""; ///< Path to a .txt file containing custom class names (one per line). If this and custom_class_labels are not set, defaults to COCO dataset.

            std::vector<std::string> custom_class_labels{}; ///< List of custom class labels in order of their IDs. If this and classmap_path are not set, defaults to COCO dataset.

            //  [optional] List of class IDs to return. All other detections will be ignored
            //  (e.g., [0] for 'person' only in COCO dataset)
            std::unordered_set<int> valid_classes; ///< List of valid class IDs to consider during post-processing. If empty, all classes will be considered.

            bool class_agnostic = false;  ///< Whether to use class-agnostic NMS (true) or class-aware NMS (false). Default is false (class-aware).
            bool fast_sigmoid = false;    ///< Whether to use fast sigmoid approximation for confidence scores. Default is false (use standard sigmoid).
            bool multi_label = false;     ///< Whether to emit one detection per class above threshold (true) or only best class per anchor (false). Default false.
            int model_id = 0; ///< The ID of the model within the running DFP to connect to. Default is 0. Useful when multiple models are mapped in the same DFP and you want to specify which one to use for post-processing.

            std::map<int, std::vector<std::string>> override_layer_mapping; ///< Optional mapping of stride values to lists of layer names for coord, conf, mask_coef, and keypoint ports (if applicable). This is used to override the automatic layer mapping based on output shapes, in case it fails. The keys should be stride values (e.g., 8, 16, 32) and the values should be lists of layer names corresponding to that stride. Use stride=0 for global feature maps such as mask_proto for segmentation. Default is an empty map, which means auto-mapping will be used.
        };

        struct YoloFinalConfig {
            float conf;
            float iou;
            int max_dets;
            bool class_agnostic;
            bool fast_sigmoid;
            bool multi_label;

            // the final list of classes, from whatever source
            std::vector<std::string> class_labels;

            // valid class IDs
            std::vector<int> valid_classes;

            int model_w;
            int model_h;

            int model_id;
            
            std::map<int, std::vector<std::string>> override_layer_mapping;
        };
    }
}
