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


#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for std::string

#include <map>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>

#include <memx/accl/MxAccl.h>
#include <memx/accl/MxAcclBase.h>

#include "memx/prepost/MxPrepost.h"

#include <opencv2/opencv.hpp>

namespace py = pybind11;
using namespace MX::Runtime;
using namespace MX::Prepost;

cv::Mat numpy_to_mat(const py::array& array) {
    py::array arr = py::array::ensure(array, py::array::c_style);
    if (!arr)
        throw std::runtime_error("Input array is not contiguous");

    py::buffer_info info = arr.request();

    if (info.ndim != 2 && info.ndim != 3)
        throw std::runtime_error("Invalid numpy array shape");

    int rows = info.shape[0];
    int cols = info.shape[1];
    int channels = (info.ndim == 3) ? info.shape[2] : 1;

    int type;
    if (info.format == py::format_descriptor<uint8_t>::format())
        type = CV_8UC(channels);
    else if (info.format == py::format_descriptor<float>::format())
        type = CV_32FC(channels);
    else
        throw std::runtime_error("Unsupported dtype");

    // zero-copy conversion
    return cv::Mat(rows, cols, type, info.ptr);
}


// python dict to std::map<int, std::vector<std::string>> (for override_layer_mapping)
std::map<int, std::vector<std::string>> dict_to_layer_mapping(const py::dict& dict) {
    std::map<int, std::vector<std::string>> mapping;
    for (auto item : dict) {
        int key = item.first.cast<int>();
        std::vector<std::string> value = item.second.cast<std::vector<std::string>>();
        mapping[key] = value;
    }
    return mapping;
}

py::array mat_to_numpy(const cv::Mat& mat) {
    // Determine dimensions
    int ndims = mat.dims;
    std::vector<size_t> shape(ndims);
    std::vector<size_t> strides(ndims);

    for (int i = 0; i < ndims; ++i) {
        shape[i] = (size_t)mat.size[i];
        strides[i] = (size_t)mat.step[i];
    }

    // If it's a standard 2D multi-channel image (H, W, C),
    // OpenCV's mat.dims is 2, but numpy expects 3 dimensions.
    if (ndims == 2 && mat.channels() > 1) {
        shape.push_back((size_t)mat.channels());
        strides.push_back((size_t)mat.elemSize1());
    }

    std::string format;
    if (mat.depth() == CV_8U)
        format = py::format_descriptor<uint8_t>::format();
    else if (mat.depth() == CV_32F)
        format = py::format_descriptor<float>::format();
    else
        throw std::runtime_error("Unsupported Mat depth");

    // zero-copy conversion
    return py::array(
            py::buffer_info(mat.data, mat.elemSize1(), format, shape.size(), shape, strides));
}

class BindMxPrepost {
  public:
    BindMxPrepost(py::object pyaccl,
                  const std::string& task,
                  float conf,
                  float iou,
                  int max_dets,
                  std::string classmap_path,
                  std::vector<std::string> custom_class_labels,
                  std::vector<int> valid_classes,
                  bool class_agnostic,
                  bool fast_sigmoid,
                  bool multi_label,
                  int model_id,
                  const py::dict& override_layer_mapping) {

        YoloUserConfig config;
        config.conf = conf;
        config.iou = iou;
        config.max_dets = max_dets;
        config.class_agnostic = class_agnostic;
        config.fast_sigmoid = fast_sigmoid;
        config.multi_label = multi_label;
        config.model_id = model_id;
        config.override_layer_mapping = dict_to_layer_mapping(override_layer_mapping);

        // convert valid_classes vector to unordered_set
        config.valid_classes =
                std::unordered_set<int>(std::make_move_iterator(valid_classes.begin()),
                                        std::make_move_iterator(valid_classes.end()));

        config.classmap_path = std::move(classmap_path);
        config.custom_class_labels = std::move(custom_class_labels);

        // TODO: check pyaccl type first
        // get PyMxAcclBase ptr from pyaccl
        py::object ptr = pyaccl.attr("get_raw_ptr")();
        uintptr_t addr = ptr.cast<uintptr_t>();
        MxAccl* accl = reinterpret_cast<MxAccl*>(addr);

        // create MxPrepost using factory method
        prepost_ = MxPrepost::create(accl, task, config);

        // get input shape from model info
        MX::Types::MxModelInfo model_info = accl->get_model_info(config.model_id);
        MX::Types::ShapeVector shape_vec = model_info.in_featuremap_shapes[0];
        for (int i = 0; i < shape_vec.size(); ++i) {
            in_shape_.push_back(shape_vec[i]);
        }
    }

    ~BindMxPrepost() {
        delete prepost_;
    }

    py::array preprocess(const py::array& arr) {

        // convert numpy to cv::Mat
        cv::Mat img = numpy_to_mat(arr);

        // call preprocess
        cv::Mat padded = prepost_->preprocess(img);

        // Reshape for model input
        int sizes[] = {in_shape_[0], in_shape_[1], in_shape_[2], in_shape_[3]};
        cv::Mat reshaped = padded.reshape(1, 4, sizes);

        // convert back to numpy
        return mat_to_numpy(reshaped);
    }

    MX::Prepost::Result postprocess(const std::vector<py::array>& ofmaps) {

        // Force user to provide original image or original shape
        throw std::runtime_error(
                "MxPrepost.postprocess(ofmaps) now requires original image or (ori_w, ori_h).\n"
                "Use:\n"
                "  postprocess(ofmaps, original_image)\n"
                "or\n"
                "  postprocess(ofmaps, ori_w, ori_h)");
    }

    // NEW overload: postprocess(ofmaps, original_image)
    MX::Prepost::Result postprocess(const std::vector<py::array>& ofmaps,
                                    const py::array& original_image) {

        // assign ofmap ptrs
        std::vector<float*> ofmap_ptrs(ofmaps.size());
        for (int i = 0; i < static_cast<int>(ofmaps.size()); ++i) {
            py::buffer_info info = ofmaps[i].request();
            float* ptr = (float*)info.ptr;
            ofmap_ptrs[i] = ptr;
        }

        // convert numpy to cv::Mat (original image)
        cv::Mat img = numpy_to_mat(original_image);

        // call postprocess
        MX::Prepost::Result result;
        prepost_->postprocess(ofmap_ptrs, result, img);  // <-- requires C++ overload
        return result;
    }

    // NEW overload: postprocess(ofmaps, ori_w, ori_h)
    MX::Prepost::Result postprocess(const std::vector<py::array>& ofmaps, int ori_h, int ori_w) {

        // assign ofmap ptrs
        std::vector<float*> ofmap_ptrs(ofmaps.size());
        for (int i = 0; i < static_cast<int>(ofmaps.size()); ++i) {
            py::buffer_info info = ofmaps[i].request();
            float* ptr = (float*)info.ptr;
            ofmap_ptrs[i] = ptr;
        }

        // call postprocess
        MX::Prepost::Result result;
        prepost_->postprocess(ofmap_ptrs, result, ori_h, ori_w);  // <-- requires C++ overload
        return result;
    }
    py::array draw(py::array& arr, const MX::Prepost::Result& result) {
        // convert numpy to cv::Mat
        cv::Mat img = numpy_to_mat(arr);

        // call draw
        prepost_->draw(img, result);

        return mat_to_numpy(img);
    }

  private:
    MxPrepost* prepost_;
    std::vector<int> in_shape_;
};

// helper to safely call import_array()
static int numpy_import_array_wrapper() {
    import_array();  // init numpy array is required in the very beginning
    return 0;
}

PYBIND11_MODULE(mxprepost, m) {
    // helper to safely call import_array(), otherwise got segfault when parsing numpy arrays
    numpy_import_array_wrapper();

    // Register C++ exception → Python exception
    py::register_exception<MX::Prepost::UnsupportedTaskError>(m, "UnsupportedTaskError");

    py::register_exception<MX::Prepost::MxError>(m, "MxError");

    // Result class
    py::class_<MX::Prepost::Result>(m, "Result",
                R"pbdoc(
                Result class for post-processing output, containing detected bounding boxes, masks, and/or keypoints.
    
                Attributes
                ----------
    
                    boxes : List[Box]
                      List of detected bounding boxes. Each box is represented as a Box object containing coordinates, confidence score, class ID, and class name.
    
                    masks : List[Mask]
                      List of detected masks. Each mask is represented as a Mask object containing the polygon coordinates, class ID, and class name.
    
                    keypoints : List[Keypoint]
                      List of detected keypoints. Each keypoint is represented as a Keypoint object containing the (x, y) coordinates and confidence score.
                )pbdoc")
            .def(py::init<>())
            .def_readwrite("boxes", &Result::boxes)
            .def_readwrite("masks", &Result::masks)
            .def_readwrite("keypoints", &Result::keypoints);

    // Box class
    py::class_<MX::Prepost::BBox>(m, "Box",
                R"pbdoc(
                Box class for representing detected bounding boxes.

                Attributes
                ----------

                    xywh : Tuple[float, float, float, float]
                      Bounding box represented as (x_center, y_center, width, height).

                    xyxy : Tuple[float, float, float, float]
                      Bounding box represented as (x_min, y_min, x_max, y_max).

                    conf : float
                      Confidence score of the detected bounding box.

                    cls_id : int
                      Class ID of the detected object.

                    cls_name : str
                      Class name as string of the detected object.
                )pbdoc")
            .def(py::init<>())
            .def_readwrite("xywh", &BBox::xywh)
            .def_readwrite("xyxy", &BBox::xyxy)
            .def_readwrite("conf", &BBox::conf)
            .def_readwrite("cls_id", &BBox::cls_id)
            .def_readwrite("cls_name", &BBox::cls_name);

    // -----------------------
    // Basic types
    // -----------------------
    py::class_<MX::Prepost::Point2f>(m, "Point2f",
                R"pbdoc(
                Point2f class for representing a point with (x, y) coordinates as floats.
    
                Attributes
                ----------
    
                    x : float
                      X coordinate of the point.
    
                    y : float
                      Y coordinate of the point.
                )pbdoc")
            // .def(py::init<>())
            .def_readwrite("x", &Point2f::x)
            .def_readwrite("y", &Point2f::y);

    // -----------------------
    // Mask
    // -----------------------
    py::class_<MX::Prepost::Mask>(m, "Mask",
                R"pbdoc(
                Mask class for representing segmentation masks.

                Attributes
                ----------

                    xys : List[Point2f]
                      List of (x, y) coordinates representing the polygon of the detected mask. Each coordinate is a Point2f object.

                    cls_id : int
                      Class ID of the detected mask.

                    cls_name : str
                      Class name as string of the detected mask.
                )pbdoc")
            // .def(py::init<>())
            .def_readwrite("xys", &Mask::xys)  // list[Point2f]
            .def_readwrite("cls_id", &Mask::cls_id)
            .def_readwrite("cls_name", &Mask::cls_name);

    // -----------------------
    // Keypoint
    // -----------------------
    py::class_<MX::Prepost::Keypoint>(m, "Keypoint",
                R"pbdoc(
                Keypoint class for representing pose estimation keypoints.

                Attributes
                ----------

                    xy : Point2f
                      (x, y) coordinates of the pose keypoint as a Point2f object.

                    conf : float
                      Confidence score of the detected pose keypoint.
                )pbdoc")
            // .def(py::init<>())
            // .def(py::init<Point2f, float>(), py::arg("xy"), py::arg("conf"))
            // .def(py::init<float, float, float>(), py::arg("x"), py::arg("y"), py::arg("conf"))
            .def_readwrite("xy", &Keypoint::xy)  // Point2f
            .def_readwrite("conf", &Keypoint::conf);

    // Prepost class
    py::class_<BindMxPrepost>(m, "MxPrepost")
            .def(py::init<py::object,
                          std::string,
                          float,
                          float,
                          int,
                          std::string,
                          std::vector<std::string>,
                          std::vector<int>,
                          bool,
                          bool,
                          bool,
                          int,
                          const py::dict&>(),
                 py::arg("accl"),
                 py::arg("task"),
                 py::arg("conf") = 0.3,
                 py::arg("iou") = 0.4,
                 py::arg("max_dets") = 300,
                 py::arg("classmap_path") = "",
                 py::arg("custom_class_labels") = py::list(),
                 py::arg("valid_classes") = py::list(),
                 py::arg("class_agnostic") = false,
                 py::arg("fast_sigmoid") = false,
                 py::arg("multi_label") = false,
                 py::arg("model_id") = 0,
                 py::arg("override_layer_mapping") = py::dict(),
                 R"pbdoc(
                 Create MxPrepost object for a specific task.

                 Parameters
                 ----------

                    accl : MxAccl
                      The MemryX accelerator object to use for post-processing. This should be an instance of the MxAccl class from the Python bindings.

                    task : str
                      Task for post-processing. Pass yolov[7|8|9|10|11]-[det|seg|pose], e.g., "yolov10-det" for detection using YOLOv10.

                    conf : float, optional
                      Confidence score threshold for post-processing. Default is 0.3.

                    iou : float, optional
                      Intersection over Union (IoU) threshold for post-processing. Default is 0.4.

                    classmap_path : str, optional
                      Path to a file containing class names, with one class name per line. If not provided, COCO classes will be used by default.

                    custom_class_labels : list of str, optional
                      List of custom class labels to use instead of loading from a file. If both classmap_path and custom_class_labels are provided, custom_class_labels will take precedence.

                    valid_classes : list of int, optional
                      List of class IDs to consider during post-processing. If not provided, all classes will be considered.

                    max_dets : int, optional
                      Maximum number of detections to keep after NMS. Default is 300.

                    class_agnostic : bool, optional
                      Use class-agnostic NMS if True. Default is False.

                    fast_sigmoid : bool, optional
                      Use fast sigmoid approximation if True. Default is False.

                    multi_label : bool, optional
                      If True, returns one detection per class above threshold per anchor. If False, returns only the best class per anchor. Default is False.

                    model_id : int, optional
                      The ID of the model to be used from the MemryX accelerator object. Default is 0. Useful when multiple models are mapped in the same DFP.

                    override_layer_mapping : dict, optional
                      A dictionary mapping stride values to lists of layer names for coord, conf, mask_coef, and keypoint ports (if applicable). This is used to override the automatic layer mapping based on output shapes, in case it fails. The keys should be stride values (e.g., 8, 16, 32) and the values should be lists of layer names corresponding to that stride. For example: {16: ["coord_layer_name", "conf_layer_name"], 32: ["coord_layer_name", "conf_layer_name"]}. Use stride=0 for global feature maps such as mask_proto for segmentation. Default is an empty dict, which means auto-mapping will be used.
                    )pbdoc")

            .def("draw",
                 &BindMxPrepost::draw,
                 py::arg("image"),
                 py::arg("result"),
                 R"pbdoc(
                 Draws the post-processing results on the input image.

                 Parameters
                 ----------

                     image : np.ndarray
                       The input image as a numpy array.

                     result : Result
                       The post-processing result containing boxes, masks, keypoints, etc., to be drawn on the image.
                
                 Returns
                 -------

                     np.ndarray
                       The image with the post-processing results drawn on it, as a numpy array.

                 )pbdoc")

            .def("preprocess",
                 &BindMxPrepost::preprocess,
                 py::arg("input_image"),
                 R"pbdoc(
                 Preprocesses the input image for model inference.

                 Parameters
                 ----------
                 
                     input_image : np.ndarray
                       The input image as a numpy array.

                 Returns
                 -------

                     np.ndarray
                       The preprocessed image ready for model input.

                 )pbdoc")

            .def("postprocess",
                 py::overload_cast<const std::vector<py::array>&, int, int>(
                         &BindMxPrepost::postprocess),
                 py::arg("ofmaps"),
                 py::arg("ori_h"),
                 py::arg("ori_w"))
            .def("postprocess",
                 py::overload_cast<const std::vector<py::array>&, const py::array&>(
                         &BindMxPrepost::postprocess),
                 py::arg("ofmaps"),
                 py::arg("original_image"),
                 R"pbdoc(
                 Postprocesses the output feature maps from the model inference and the original image shape


                 Parameters
                 ----------

                     ofmaps : List[np.ndarray]
                       A list of output feature maps from the model inference, as numpy arrays.

                     ori_h : int  (function overload 1 only)
                       The original height of the input image. Not the model's height -- use the stream's original resolution.

                     ori_w : int  (function overload 1 only)
                       The original width of the input image. Not the model's width -- use the stream's original resolution.

                     original_image : np.ndarray  (function overload 2 only)
                       The original input image as a numpy array. This is used to provide the original shape context for post-processing only.


                 Returns
                 -------

                      Result
                        The post-processing result containing boxes, masks, keypoints, etc.

                )pbdoc");
}
