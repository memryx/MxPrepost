# MxPrepost

Unified, high-performance **YOLO pre-processing and post-processing** for [MemryX MXA](https://developer.memryx.com) applications — detection, segmentation, and pose estimation across YOLOv7–v26.

`MxPrepost` handles:

- **Pre-processing:** frame preparation (resize, normalization, layout for the DFP).
- **Post-processing:** output decoding, NMS, class filtering, and optional visualization.

**License:** [MIT](LICENSE)  
**Docs:** [MxPrepost usage guide](https://developer.memryx.com/runtime/mxprepost.html)

---

## Supported models

| Model | Detection | Segmentation | Pose |
|-------|-----------|--------------|------|
| YOLOv7 | ✅ | — | — |
| YOLOv8 | ✅ | ✅ | ✅ |
| YOLOv9 | ✅ | — | — |
| YOLOv10 | ✅ | — | — |
| YOLOv11 | ✅ | ✅ | ✅ |
| YOLOv26 | ✅ | ✅ | ✅ |

Also supported: **custom class maps**, **custom input resolutions** (not limited to 640×640).

---

## Dependency & Compatibility

| Requirement | Notes |
|-------------|--------|
| **OS** | Linux (x86_64 or aarch64). |
| **OpenCV** | Ubuntu/Debian: `sudo apt install libopencv-dev` |
| **MemryX Runtime / SDK** | Supporting SDK **2.2.+** - [installation guide](https://developer.memryx.com/get_started/install_software.html) |
 
> Legacy bindings are **not** supported: `SyncAccl`, `AsyncAccl`, `MultistreamAsyncAccl`.

---

## Installation

### Option 1: Install prebuilt package (recommended)

Install the prebuilt package from the MemryX package index:

```bash
pip3 install --extra-index-url https://developer.memryx.com/pip mxprepost
```

Or install the latest release directly from GitHub:

```bash
pip install "git+https://github.com/memryx/MxPrepost.git"
```

No local compilation is required.

### Option 2: Build from source

Activate the MemryX SDK environment:

```bash
source ~/.mx/bin/activate
```

#### Build

```bash
git clone --recurse-submodules https://github.com/memryx/MxPrepost.git
cd MxPrepost

sh build.sh
```

> If `build.sh` requires elevated permissions on your system, run the build commands manually instead:
>
> ```bash
> mkdir -p build && cd build
> cmake ..
> make -j$(nproc)
>
> cd ../pymodule
> mkdir -p build && cd build
> cmake ..
> make -j$(nproc)
> ```
This builds:

- `libmxprepost.so`
- Python extension:
  `pymodule/build/mxprepost.cpython-*.so`

### Local Development

After building, link the generated Python module into your application:

```bash
ln -sfv /your/absolute/path/to/MxPrepost/pymodule/build/mxprepost.cpython-*.so
```


---

## Python Usage (MXA integration)

1. Create `mxapi.MxAccl`
2. Connect input/output callbacks
3. Create `mxprepost.MxPrepost(accl=..., task=...)`
4. In callbacks:
   - **Input:** `preprocess(frame)`
   - **Output:** `postprocess(mxa_output, ori_height, ori_width)` or `postprocess(mxa_output, ori_frame)`
   - **Optional:** `draw(frame, result)`

Reference implementation: [`samples/python/run.py`](samples/python/run.py).

```python
import cv2
import mxprepost
from memryx import mxapi


class App:
    def __init__(self, dfp_path, video_path):
        self.cap = cv2.VideoCapture(video_path)

        # Store original frame dimensions
        self.ori_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.ori_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create MXA accelerator
        self.accl = mxapi.MxAccl(dfp_path, [0], [False, False], False)

		#---------------------------------------------------------------
		#-------------------------- MxPrepost --------------------------
		#---------------------------------------------------------------
        # Initialize MxPrepost
        self.pp = mxprepost.MxPrepost(
            accl=self.accl,
            task="yolov8-det",  # e.g. "yolov8-det", "yolov11-seg", "yolov26-pose"
        )

        self.accl.connect_stream(self.in_callback, self.out_callback, stream_id=0)

    def in_callback(self, stream_id):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return self.pp.preprocess(frame)

    def out_callback(self, mxa_output, stream_id):
		self.pp.postprocess(mxa_output,ori_image)
		result = self.pp.postprocess(
            mxa_output,
            self.ori_height,
            self.ori_width,
        ) # or self.pp.postprocess(mxa_output, self.ori_image)

        # Access outputs (depending on task)
		# ---------------------------
        # Detection (Bounding Boxes)
        # ---------------------------
        if result.boxes is not None:
            for box in result.boxes:
                print("xyxy:", box.xyxy)        # [x1, y1, x2, y2]
                print("xywh:", box.xywh)        # [x_center, y_center, w, h]
                print("conf:", box.conf)        # confidence score
                print("cls_id:", box.cls_id)    # class index
                print("cls_name:", box.cls_name)# class label string
                print("----")

        # ---------------------------
        # Segmentation (Masks)
        # ---------------------------
        if result.masks is not None:
            for mask in result.masks:
                # mask is a polygon (list of Point2f), not a binary HxW array
                pts = [(p.x, p.y) for p in mask.xys]
                print("mask cls_id:", int(mask.cls_id), "points:", len(pts))

        # ---------------------------
        # Pose Estimation (Keypoints)
        # ---------------------------
        if result.keypoints is not None:
            for det_id, kps in enumerate(result.keypoints):
                print(f"detection {det_id} keypoints:", len(kps))
                # each kp has kp.xy (Point2f) and kp.conf
                for kp_id, kp in enumerate(kps):
                    print(f"  kp[{kp_id}] = (x={kp.xy.x:.2f}, y={kp.xy.y:.2f}, conf={kp.conf:.3f})")
                print("----")

        # Optional: draw detections
        # annotated = self.pp.draw(frame, result)
```
### Configuration

```python
prepost = mxprepost.MxPrepost(
    accl=accl,                              # [required] mxapi.MxAccl instance
    task="yolov8-det",                      # [required] yolov<n>-[det|seg|pose]
    conf=0.3,                               # [optional] default 0.3
    iou=0.4,                                # [optional] default 0.4
    classmap_path="/path/to/classmap.txt",  # [optional] one class name per line; default COCO
    custom_class_labels=["person", "car"],  # [optional] list instead of classmap file
    valid_classes=[],                       # [optional] e.g. [0] for person-only on COCO
    model_id=0,                             # [optional] when multiple models share one DFP
    class_agnostic=False,                   # [optional] class-agnostic NMS if True
    fast_sigmoid=False,                     # [optional] faster approximate sigmoid
    multi_label=False,                      # [optional] If True, return one detection per class above threshold per anchor. If False, keep only the best class per anchor.
    max_dets=300,                           # [optional] max boxes after NMS
    override_layer_mapping={},              # [optional] override stride layer -> DFP port map
)
```

`task` examples: `yolov8-det`, `yolov11-seg`, `yolov26-pose`.  
Set `model_id` only when multiple models are compiled into the same DFP.

---

## C++ Usage (MXA integration)

Same callback pattern as Python:

1. `MX::Runtime::MxAccl`
2. `MxPrepost::create(&accl, task, cfg)` (or `create_safe`)
3. `preprocess(frame)` in the input callback
4. `postprocess(ofmap_ptrs, result, ori_h, ori_w)` or `postprocess(..., original_frame)` in the output callback
5. Optional `draw(frame, result)`

```cpp
#include <memx/accl/MxAccl.h>
#include <memx/prepost/MxPrepost.h>
#include <opencv2/opencv.hpp>

using namespace MX::Runtime;

MxAccl accl{dfp_path, {0}, {false, false}, false};

//---------------------------------------------------------------
//-------------------------- MxPrepost --------------------------
//---------------------------------------------------------------
// Create MxPrepost (task examples: "yolov8-det", "yolov8-seg", "yolov11-pose")
YoloUserConfig cfg; // inits with the defaults for COCO
cfg.conf = 0.3f;
cfg.iou  = 0.4f;
cfg.max_dets = 300;
std::unique_ptr<MxPrepost> pp{MxPrepost::create(&accl, task, cfg)};

// Get original dimensions once (from your cv::VideoCapture)
int ori_w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
int ori_h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);

// --- Input callback: capture + preprocess ---
auto in_cb = [&](std::vector<const MX::Types::FeatureMap*> dst, int stream_id) -> bool {
    cv::Mat frame;
    if (!cap.read(frame)) return false;

    //---------------------------------------------------------------
	//-------------------------- MxPrepost --------------------------
	//---------------------------------------------------------------
    std::unique_ptr<MxPrepost> pp;

    // ===========================
    // Method 1: Throwing create()
    // ===========================
    try {
        pp.reset(MxPrepost::create(&accl, task, cfg));   // may throw (e.g., unsupported task)
    } catch (const std::exception& e) {
        std::cerr << "Failed to create MxPrepost: " << e.what() << "\n";
        return 1; // or handle error appropriately
    }

    // ==================================
    // Method 2: No-throw create_safe()
    // ==================================
    // std::string err;
    // if (!MxPrepost::create_safe(&accl, task, cfg, pp, err)) {
    //     std::cerr << "Failed to create MxPrepost: " << err << "\n";
    //     return 1; // or handle error appropriately
    // }

    dst[0]->set_data(reinterpret_cast<float*>(input.data));
    return true;
};

// --- Output callback: postprocess
auto out_cb = [&](std::vector<const MX::Types::FeatureMap*> outs, int stream_id) -> bool {
    // Copy accelerator outputs into your own float buffers, then build float* vector:
    
    //---------------------------------------------------------------
	//-------------------------- MxPrepost --------------------------
	//---------------------------------------------------------------
    Result result;
    pp->postprocess(ofmap_ptrs, result, ori_h, ori_w);   // or: pp->postprocess(ofmap_ptrs, result, original_frame)

    // Access outputs (depending on task)
    // ---------------------------
    // Detection (Bounding Boxes)
    // ---------------------------
    for (const auto& box : result.boxes) {
        // box.xyxy = [x1,y1,x2,y2], box.xywh = [xc,yc,w,h]
        // box.conf, box.cls_id, box.cls_name
    }

    // ---------------------------
    // Segmentation (polygons)
    // ---------------------------
    for (const auto& mask : result.masks) {
        // mask.xys is a polygon: vector<Point2f>
        // mask.cls_id
    }

    // ---------------------------
    // Pose (keypoints)
    // ---------------------------
    for (size_t det_id = 0; det_id < result.keypoints.size(); ++det_id) {
        const auto& kps = result.keypoints[det_id]; // vector<Keypoint>
        // each kp has kp.xy (Point2f) and kp.conf
    }

    // Optional visualization
    // pp->draw(frame, result);

    return true;
};

accl.connect_stream(in_cb, out_cb, 0);
accl.start();
accl.wait();
```

---


## For Sample Codes Usage

### Python sample (`samples/python`)

```bash
# Webcam
python run.py -d /path/to/model.dfp -t yolov8-det
```

```bash
# Video file
python run.py -d /path/to/model.dfp -t yolov8-det --video_paths videos/sample.mp4
```
```bash
# Multiple streams
python run.py -d /path/to/model.dfp -t yolov8-det --video_paths /dev/video0 videos/sample.mp4
```

```bash
# Benchmark (no display)
python run.py -d /path/to/model.dfp -t yolov8-det --video_paths /dev/video0 --no-show
```
| Argument | Description | Default |
|----------|-------------|---------|
| `-d`, `--dfp` | Path to compiled `.dfp` | **Required** |
| `-t`, `--task` | Task string (`yolov8-det`, `yolov11-seg`, …) | **Required** |
| `--video_paths` | Camera device or video file(s) | `/dev/video0` |
| `--no-show` | Disable OpenCV window | display on |

### C++ Sample (`samples/cpp`)

Similar to python sample

```bash
./build/samples/cpp/run \
  -d /path/to/model.dfp \
  -t yolov8-det \
  --video_paths /dev/video0
```
