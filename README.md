# 🦺 PPE Detection System

> Real-time Personal Protective Equipment detection on **NVIDIA Jetson Orin Nano** using YOLOv8m + TensorRT FP16.

[![Python](https://img.shields.io/badge/Python-3.8-blue)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8m-Ultralytics-red)](https://ultralytics.com)
[![TensorRT](https://img.shields.io/badge/TensorRT-8.5-green)](https://developer.nvidia.com/tensorrt)
[![JetPack](https://img.shields.io/badge/JetPack-5.1.3-orange)](https://developer.nvidia.com/embedded/jetpack)

---

## 📋 Overview

This system continuously monitors whether workers are wearing required PPE — **helmet**, **high-visibility vest**, and **protective gloves** — and triggers real-time alerts when violations are detected.

| Metric | Result | Target |
|--------|--------|--------|
| mAP @ 0.50 | **0.9149** | ≥ 0.80 ✅ |
| Precision | **0.9354** | ≥ 0.85 ✅ |
| Recall | **0.8579** | ≥ 0.80 ✅ |
| F1-Score | **0.8950** | ≥ 0.82 ✅ |
| Inference latency | **~22 ms** | < 30 ms ✅ |
| Throughput | **~45 FPS** | > 30 FPS ✅ |

---

## 🗂️ Project Structure

```
ppe_detection/
├── src/
│   ├── train.py                  # YOLOv8 fine-tuning script
│   ├── export_tensorrt.py        # ONNX + TensorRT export
│   ├── tensorrt_infer.py         # TensorRT inference engine wrapper
│   ├── alert_logger.py           # Alert channels + audit logging
│   └── inference_gstreamer.py    # Main inference + MJPEG stream
├── dashboard/
│   ├── main.py                   # FastAPI backend + WebSocket server
│   └── templates/
│       └── index.html            # Supervisor dashboard frontend
├── configs/
│   ├── deploy.yaml               # Runtime configuration
│   └── ppe_dataset.yaml          # Dataset paths and class names
├── Dockerfile                    # Jetson-optimised container
├── requirements.txt              # Python dependencies
└── README.md
```

---

## ⚙️ Hardware & Software

| Component | Details |
|-----------|---------|
| Edge Device | NVIDIA Jetson Orin Nano Developer Kit |
| JetPack | 5.1.3 (CUDA 11.4 · cuDNN 8.6 · TRT 8.5) |
| Model | YOLOv8m (fine-tuned) |
| Inference | TensorRT FP16 engine |
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/CSS/JS + WebSocket |
| Camera | USB/CSI via V4L2 |

---

## 🚀 Quick Start

### 1. Clone and install

```bash
git clone https://github.com/HEX027/ppe_detection.git
cd ppe_detection
pip3 install -r requirements.txt
pip3 install "uvicorn[standard]" fastapi aiofiles
```

### 2. Build TensorRT engine (on Jetson)

```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx=models/onnx/ppe_model.onnx \
    --saveEngine=models/tensorrt/ppe_fp16.engine \
    --fp16 --workspace=2048
```

### 3. Run the system

```bash
python3 dashboard/main.py
```

Open browser at `http://localhost:8000`

---

## 🏋️ Training (Google Colab)

```python
from ultralytics import YOLO

model = YOLO("yolov8m.pt")
model.train(
    data    = "configs/ppe_dataset.yaml",
    epochs  = 40,
    imgsz   = 640,
    batch   = 32,
    device  = 0,
    mosaic  = 1.0,
    fliplr  = 0.5,
    hsv_h   = 0.015,
    hsv_s   = 0.7,
    scale   = 0.5,
)
```

Dataset: [Roboflow PPE Dataset v22](https://universe.roboflow.com/haya-halhr/ppe-dataset-orfty) — 11,342 images

---

## 📊 Classes

| ID | Class | AP @ 0.50 |
|----|-------|-----------|
| 0 | Helmet | 0.9683 |
| 1 | Vest | 0.9408 |
| 2 | Gloves | 0.8357 |

---

## 🔔 Alert Channels

| Channel | Description |
|---------|-------------|
| On-screen | Red banner with missing PPE names + bounding boxes |
| Buzzer | WAV file via ALSA with cooldown |
| Webhook | HTTP POST JSON to supervisor endpoint |

---

## 📝 Audit Log Format

```json
{
  "timestamp": "2026-05-19T01:21:45Z",
  "zone_id": "zone_A",
  "zone_label": "Zone A",
  "missing_ppe": ["gloves"],
  "frame_index": 155,
  "snapshot_path": "/logs/snapshots/zone_A_..._00000155.jpg",
  "detections": [
    {"class_name": "helmet", "confidence": 0.741, "bbox": [120, 45, 280, 190]},
    {"class_name": "vest",   "confidence": 0.823, "bbox": [90, 180, 350, 520]}
  ]
}
```

---

## 🐳 Docker

```bash
docker build -t ppe_detection:v1 .

docker run -d \
  --name ppe_system \
  --runtime nvidia \
  --privileged \
  --device /dev/video0:/dev/video0 \
  -v $(pwd)/models:/workspace/ppe_detection/models \
  -v $(pwd)/logs:/workspace/ppe_detection/logs \
  -p 8000:8000 \
  ppe_detection:v1
```

---

## 📁 Logs & Snapshots

```
logs/
├── events/
│   └── ppe_violations_YYYY-MM-DD.jsonl   # Daily audit log
└── snapshots/
    └── zone_A_2026-05-19_*.jpg           # Violation frame captures
```

---

## 📄 License

This project is for academic and research purposes.

---

*Built on NVIDIA Jetson Orin Nano · YOLOv8 · TensorRT · FastAPI*
