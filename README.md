# SmartVision Lite — Real-Time Object Detection

SmartVision Lite is a beginner-friendly object-detection project built with
Ultralytics YOLO, PyTorch, OpenCV, and Streamlit. It provides one approachable
interface for detecting objects in images, uploaded videos, and a live webcam
feed, plus a standalone desktop webcam demo and a Colab training notebook.

> Detection output is probabilistic. Do not use this project as the sole basis
> for safety-critical, medical, legal, or surveillance decisions.

## Features

- Polished Streamlit home page with a guided workflow
- Image upload, annotation, total and per-class counts, confidence details, and
  PNG/CSV/JSON downloads
- Frame-by-frame video processing with progress, average FPS, summary metrics,
  and a downloadable annotated video when a compatible codec is available
- Browser webcam detection through WebRTC with a clear optional-dependency
  fallback
- Standalone OpenCV webcam demo that always releases the camera
- Pretrained `yolo11n.pt` or local custom `best.pt` selection
- Confidence, IoU, image-size, and Auto/CPU/CUDA controls
- Defensive handling for invalid media, corrupt videos, missing cameras,
  missing models, failed downloads, and unavailable CUDA
- Modular, typed detection code with deterministic tests that require neither
  a model download nor webcam hardware
- Google Colab custom-training notebook and GitHub Actions CI

## Technology stack

| Layer | Technology |
| --- | --- |
| Detection | Ultralytics YOLO, PyTorch |
| Images/video | OpenCV, NumPy, Pillow |
| Web app | Streamlit, streamlit-webrtc |
| Tabular exports | pandas |
| Tests and CI | pytest, GitHub Actions |

## Architecture

```text
Streamlit / desktop webcam
            |
            v
 ImageProcessor / VideoProcessor
            |
            v
       ObjectDetector
            |
            v
 Ultralytics YOLO + PyTorch device
```

The UI never fabricates results. `ObjectDetector` converts Ultralytics output
into small typed records; processors handle media decoding and encoding; metric
and export helpers remain independent and easy to test.

## Project structure

```text
SmartVision-Lite/
|-- app.py                       # Streamlit application
|-- webcam_demo.py               # Desktop OpenCV webcam demo
|-- config/                      # Defaults and supported options
|-- detection/                   # Detector, processors, metrics, utilities
|-- models/                      # Local weights (weights are Git-ignored)
|-- sample_data/                 # Local media (media is Git-ignored)
|-- outputs/                     # Generated files (contents are Git-ignored)
|-- training/                    # Colab notebook and dataset example
|-- tests/                       # Hardware/network-independent unit tests
|-- .github/workflows/           # Continuous integration
|-- .streamlit/                  # Streamlit theme and server settings
|-- requirements.txt
`-- pyproject.toml
```

## Windows installation

Python 3.10–3.12 and Git are recommended. In PowerShell:

```powershell
git clone https://github.com/mohitkhairanar-max/SmartVision-Lite.git
cd SmartVision-Lite
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, either allow locally created scripts for your
user account or call the environment directly:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The first pretrained-model run may download `yolo11n.pt` from Ultralytics. A
failed or blocked download is reported in the UI; model files are not committed.

## Run the Streamlit app

```powershell
streamlit run app.py
```

Open `http://localhost:8501`, choose a mode, select settings in the sidebar,
then supply an image, video, or camera permission. Use **Stop** in the WebRTC
widget before leaving the live page.

## Run the desktop webcam demo

```powershell
python webcam_demo.py
```

Press `q` in the OpenCV window to close it. Optional arguments are available:

```powershell
python webcam_demo.py --model models/best.pt --confidence 0.35 --device cpu
```

The desktop script reads camera index `0` by default. It prints a clear error and
returns a non-zero exit code when that camera cannot be opened.

## Use a custom model

1. Put a trained checkpoint at `models/best.pt` (or another local `.pt` path).
2. Select **Custom model** in the Streamlit sidebar.
3. Enter the path and apply detection settings.

An invalid, missing, or incompatible checkpoint is rejected with a useful
message. Never load model files from an untrusted source: PyTorch checkpoints
may contain executable serialized content. All `*.pt` files are Git-ignored.

## Train a custom model

Open `training/custom_training.ipynb` in Google Colab. The notebook installs
Ultralytics, accepts an uploaded/extracted YOLO dataset or an existing Colab
path, validates `dataset.yaml`, trains a nano model, validates it, explains the
reported precision/recall/mAP values, shows the confusion matrix, and downloads
`best.pt`.

Roboflow-exported YOLO datasets work after manual download/export. No Roboflow
API key is included or requested by the notebook. See
`training/dataset.example.yaml` for the expected configuration.

### Dataset format

Each image has a matching `.txt` label file. Each row is:

```text
class_id x_center y_center width height
```

Coordinates are normalized to the range 0–1. The YAML file maps numeric class
IDs to names and points to the train, validation, and optional test splits.

## Understanding the metrics

- **Confidence** estimates how strongly the model supports an individual
  prediction after filtering.
- **Precision** is the fraction of predicted positives that match ground truth.
- **Recall** is the fraction of ground-truth objects the model finds.
- **mAP50** is mean average precision using an IoU match threshold of 0.50.
- **mAP50–95** averages mAP over IoU thresholds 0.50 through 0.95 and is more
  demanding.
- **Inference time** measures the detector call for one image or frame; decoding,
  rendering, transfer, and video encoding can add more latency.
- **FPS** is processed frames divided by measured elapsed wall time. The displayed
  average can vary with the model, input size, hardware, warm-up, and encoder.

Image counts describe detections in that image. Video totals are frame-level
detections summed across frames; they are **not unique-object counts**, because
tracking is outside this lite project's scope.

## Benchmark FPS responsibly

1. Use the same model, image size, thresholds, device, and test video.
2. Allow a short warm-up, especially on CUDA.
3. Measure enough frames to smooth startup effects.
4. State whether decoding, annotation, and encoding are included.
5. Report hardware and software versions with the measured average.

This repository intentionally makes no fixed accuracy or FPS claim.

## CPU and CUDA

- **Auto** selects CUDA when PyTorch reports it available, otherwise CPU.
- **CPU** is portable but may be slower, especially for large inputs.
- **CUDA** requires a compatible NVIDIA driver and a CUDA-enabled PyTorch build.
  If CUDA is requested but unavailable, SmartVision Lite reports the problem and
  safely falls back to CPU rather than crashing.

Check your environment with:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Tests

Unit tests inject lightweight fake models, so CI does not download weights,
open a GUI, or access a webcam:

```powershell
python -m compileall -q app.py webcam_demo.py config detection tests
python -m pytest
```

A real-model smoke test requires network access the first time and runs without
opening a window or camera:

```powershell
python scripts/smoke_test.py --device cpu
```

## Screenshots

Screenshots will be added after capturing the app on representative hardware.
No mock detection result is shown here.

- Home page — placeholder
- Image detection and exports — placeholder
- Video progress and summary — placeholder
- Live webcam mode — placeholder

## Troubleshooting

| Problem | Suggested fix |
| --- | --- |
| `python` or `streamlit` is not recognized | Activate `.venv`, or call `.venv\Scripts\python.exe -m streamlit run app.py`. |
| Model download fails | Check internet/proxy settings, retry, or provide a trusted local `best.pt`. |
| Custom model cannot load | Verify the path, `.pt` suffix, permissions, and Ultralytics compatibility. |
| CUDA is unavailable | Choose Auto/CPU or install the correct NVIDIA driver and PyTorch build. |
| Browser camera is blank | Allow camera permission, close other camera apps, and use HTTPS when accessing a remote host. |
| Desktop camera cannot open | Try another `--camera` index and close programs using the webcam. |
| Uploaded video is rejected | Confirm MP4/AVI/MOV format and re-encode a corrupted or unsupported codec. |
| Processed video will not preview | Download it and use a compatible player; browser codec support varies. |
| No objects are detected | Lower confidence carefully, try a larger image size, or use a model trained for those classes. |

## Known limitations

- The pretrained COCO model recognizes its training classes, not every possible
  object.
- Webcam availability and performance depend on browser/OS permissions and
  hardware; automated CI cannot physically verify a camera.
- Video encoding is platform- and codec-dependent.
- The app performs detection, not multi-frame tracking or unique-person counting.
- Very large videos require time, memory, and temporary disk space.

## Future improvements

- Optional object tracking and unique-object analytics
- Batched and asynchronous video inference
- Model comparison and reproducible benchmark reports
- Segmentation and pose-estimation modes
- Container images and hosted deployment guidance

## License and author

Released under the [MIT License](LICENSE).

**Mohit Khairanar** — [@mohitkhairanar-max](https://github.com/mohitkhairanar-max)
