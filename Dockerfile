# ============================================================
# PPE Detection System - Dockerfile for NVIDIA Jetson Orin Nano
# Base: NVIDIA L4T PyTorch image (JetPack 5.x / CUDA 11.4)
# ============================================================

FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

# --- System metadata ---
LABEL maintainer="PPE Detection System"
LABEL description="Real-time PPE detection on NVIDIA Jetson Orin Nano"

# --- Avoid interactive prompts during apt installs ---
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── System dependencies ────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build tools
    build-essential cmake git pkg-config \
    # Python
    python3-pip python3-dev python3-setuptools python3-wheel \
    # OpenCV system libs (camera, video codec)
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-good1.0-dev gstreamer1.0-plugins-bad \
    gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x \
    gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 \
    # Audio (buzzer alerts via ALSA)
    alsa-utils libasound2-dev \
    # Networking
    libcurl4-openssl-dev \
    # Image/video codec support
    libjpeg-dev libpng-dev libtiff-dev \
    ffmpeg v4l-utils \
    # Utilities
    wget curl unzip htop nano \
    && rm -rf /var/lib/apt/lists/*

# ── Python package upgrades ────────────────────────────────
RUN pip3 install --upgrade pip setuptools wheel

# ── Core ML / Vision dependencies ─────────────────────────
# Note: torch & torchvision are already in the L4T base image.
# We install ONNX, TensorRT Python bindings, and YOLOv8 on top.
RUN pip3 install --no-cache-dir \
    # YOLO framework (Ultralytics YOLOv8)
    ultralytics==8.2.0 \
    # ONNX export
    onnx==1.16.0 \
    onnxruntime-gpu==1.18.0 \
    onnxsim==0.4.36 \
    # TensorRT Python bindings (pre-installed on JetPack; listed for clarity)
    # tensorrt  ← provided by JetPack; do NOT pip-install separately
    # Data science / training utilities
    numpy==1.24.4 \
    scipy==1.11.4 \
    pandas==2.0.3 \
    matplotlib==3.7.5 \
    seaborn==0.13.2 \
    scikit-learn==1.3.2 \
    pycocotools==2.0.8 \
    # OpenCV (headless — GUI handled by GStreamer/DeepStream)
    opencv-python-headless==4.9.0.80 \
    # Augmentation library
    albumentations==1.4.8 \
    # Roboflow dataset download
    roboflow==1.1.34 \
    # Alert / notification
    requests==2.31.0 \
    playsound==1.3.0 \
    # Logging & monitoring
    loguru==0.7.2 \
    tqdm==4.66.4 \
    pyyaml==6.0.1 \
    python-dotenv==1.0.1

# ── GStreamer Python bindings ──────────────────────────────
RUN pip3 install --no-cache-dir \
    PyGObject==3.44.1 \
    gst-python || true   # falls back to system package

# ── TensorRT Python bindings (JetPack path) ───────────────
# On Jetson, tensorrt is installed via apt/JetPack, not pip.
# Add the JetPack TensorRT site-packages to PYTHONPATH.
ENV PYTHONPATH="${PYTHONPATH}:/usr/lib/python3/dist-packages:/usr/local/lib/python3.8/dist-packages"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/lib/aarch64-linux-gnu/tegra:/usr/local/cuda/lib64"

# ── Working directory & project structure ─────────────────
WORKDIR /workspace/ppe_detection

# Copy project files
COPY . .

# Create required runtime directories
RUN mkdir -p \
    data/raw data/processed data/augmented \
    models/weights models/onnx models/tensorrt \
    logs/snapshots logs/events \
    configs

# ── Expose ports (network alert webhook receiver) ─────────
EXPOSE 8080

# ── Healthcheck ────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import torch; import ultralytics; print('OK')" || exit 1

# ── Default command ────────────────────────────────────────
CMD ["python3", "src/inference_gstreamer.py", "--config", "configs/deploy.yaml"]
