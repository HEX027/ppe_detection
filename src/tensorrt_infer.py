#!/usr/bin/env python3
"""
tensorrt_infer.py — TensorRT Engine Inference Wrapper
==============================================================
Loads a serialised TensorRT .engine file, allocates pinned/device
memory, and exposes a predict() method that returns a list of
Detection objects.

Fallback: if TensorRT is unavailable, the wrapper automatically
falls back to ONNX Runtime for development / testing.
==============================================================
"""

import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.alert_logger import Detection


# ─────────────────────────────────────────────────────────
# TensorRT Inference Engine
# ─────────────────────────────────────────────────────────
class TRTInferenceEngine:
    """
    Wraps a serialised TensorRT engine for single-batch inference.

    Usage
    ─────
    engine = TRTInferenceEngine(
        engine_path="models/tensorrt/ppe_fp16.engine",
        class_names=["helmet", "vest", "gloves"],
        conf_thresh=0.50,
        iou_thresh=0.45,
        input_size=(640, 640),
    )
    detections = engine.predict(bgr_frame)
    """

    def __init__(
        self,
        engine_path:  str,
        class_names:  List[str],
        conf_thresh:  float = 0.50,
        iou_thresh:   float = 0.45,
        input_size:   Tuple[int, int] = (640, 640),
    ):
        self.class_names = class_names
        self.conf_thresh = conf_thresh
        self.iou_thresh  = iou_thresh
        self.input_size  = input_size   # (width, height)
        self._engine_path = engine_path

        self._use_trt = self._try_load_trt(engine_path)

        if not self._use_trt:
            logger.warning(
                "TensorRT unavailable — falling back to ONNX Runtime"
            )
            onnx_path = engine_path.replace(".engine", ".onnx").replace(
                "tensorrt", "onnx"
            )
            self._init_onnx(onnx_path)

    # ── TensorRT initialisation ───────────────────────────
    def _try_load_trt(self, engine_path: str) -> bool:
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401
        except ImportError:
            return False

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(TRT_LOGGER, "")
        runtime = trt.Runtime(TRT_LOGGER)

        with open(engine_path, "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())

        self._context = self._engine.create_execution_context()
        self._cuda    = cuda

        # Allocate I/O buffers
        self._bindings: List = []
        self._host_inputs:  List[np.ndarray] = []
        self._host_outputs: List[np.ndarray] = []
        self._cuda_inputs:  List = []
        self._cuda_outputs: List = []

        for i in range(self._engine.num_bindings):
            shape = tuple(self._engine.get_binding_shape(i))
            dtype = trt.nptype(self._engine.get_binding_dtype(i))
            host_mem   = cuda.pagelocked_empty(shape, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self._bindings.append(int(device_mem))

            if self._engine.binding_is_input(i):
                self._host_inputs.append(host_mem)
                self._cuda_inputs.append(device_mem)
            else:
                self._host_outputs.append(host_mem)
                self._cuda_outputs.append(device_mem)

        self._stream = cuda.Stream()
        logger.success(f"TensorRT engine loaded: {engine_path}")
        return True

    # ── ONNX Runtime fallback ─────────────────────────────
    def _init_onnx(self, onnx_path: str) -> None:
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._ort_session = ort.InferenceSession(onnx_path, providers=providers)
        self._ort_input_name = self._ort_session.get_inputs()[0].name
        logger.success(f"ONNX Runtime session loaded: {onnx_path}")

    # ── Pre-processing ────────────────────────────────────
    def _preprocess(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        """
        Resize → BGR→RGB → normalise [0,1] → NCHW float32.

        Returns
        ───────
        blob       : np.ndarray  shape [1, 3, H, W]
        scale      : float       resize scale factor
        pad_top    : int         vertical letterbox padding (px)
        pad_left   : int         horizontal letterbox padding (px)
        """
        target_w, target_h = self.input_size
        orig_h, orig_w = frame_bgr.shape[:2]

        # Letterbox resize (preserves aspect ratio)
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to target size
        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        pad_top  = (target_h - new_h) // 2
        pad_left = (target_w - new_w) // 2
        canvas[pad_top: pad_top + new_h, pad_left: pad_left + new_w] = resized

        # BGR → RGB, normalise, NCHW
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        blob = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]

        return blob, scale, pad_top, pad_left

    # ── Post-processing ────────────────────────────────────
    def _postprocess(
        self,
        raw_output: np.ndarray,
        orig_shape: Tuple[int, int],
        scale: float,
        pad_top: int,
        pad_left: int,
    ) -> List[Detection]:
        """
        Decode YOLOv8 output tensor → Detection list.

        YOLOv8 ONNX output shape: [1, 4+nc, num_anchors]
        Where 4 = cx, cy, w, h and nc = num classes.

        Applies:
          • confidence thresholding
          • Non-Maximum Suppression (NMS)
          • Letterbox coordinate de-padding
        """
        orig_h, orig_w = orig_shape

        # output shape may be [1, 4+nc, 8400] — transpose to [8400, 4+nc]
        preds = raw_output[0]   # [4+nc, 8400] or [8400, 4+nc]
        if preds.shape[0] < preds.shape[1]:
            preds = preds.T     # → [8400, 4+nc]

        # Split boxes and class scores
        boxes  = preds[:, :4]                    # cx, cy, w, h
        scores = preds[:, 4:]                    # [8400, nc]

        # Class confidence = max class score
        class_ids = np.argmax(scores, axis=1)    # [8400]
        confs     = scores[np.arange(len(scores)), class_ids]  # [8400]

        # Filter by confidence threshold
        mask  = confs >= self.conf_thresh
        boxes = boxes[mask]
        confs = confs[mask]
        class_ids = class_ids[mask]

        if len(boxes) == 0:
            return []

        # cx,cy,w,h → x1,y1,x2,y2
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # NMS per class
        detections: List[Detection] = []
        for cls_id in np.unique(class_ids):
            cls_mask   = class_ids == cls_id
            cls_boxes  = xyxy[cls_mask]
            cls_confs  = confs[cls_mask]

            # OpenCV NMSBoxes expects float lists
            nms_idxs = cv2.dnn.NMSBoxes(
                cls_boxes.tolist(),
                cls_confs.tolist(),
                self.conf_thresh,
                self.iou_thresh,
            )
            if len(nms_idxs) == 0:
                continue

            for i in nms_idxs.flatten():
                bx1, by1, bx2, by2 = cls_boxes[i]

                # Remove letterbox padding, rescale to original coords
                bx1 = max(0, (bx1 - pad_left) / scale)
                by1 = max(0, (by1 - pad_top)  / scale)
                bx2 = min(orig_w, (bx2 - pad_left) / scale)
                by2 = min(orig_h, (by2 - pad_top)  / scale)

                detections.append(Detection(
                    class_id=int(cls_id),
                    class_name=self.class_names[int(cls_id)],
                    confidence=float(cls_confs[i]),
                    bbox_xyxy=(int(bx1), int(by1), int(bx2), int(by2)),
                ))

        return detections

    # ── Run inference ─────────────────────────────────────
    def predict(self, frame_bgr: np.ndarray) -> List[Detection]:
        """
        Run inference on a single BGR frame.

        Returns a list of Detection objects with pixel coordinates
        relative to the original (un-resized) frame.
        """
        blob, scale, pad_top, pad_left = self._preprocess(frame_bgr)
        orig_shape = frame_bgr.shape[:2]

        if self._use_trt:
            raw_output = self._trt_infer(blob)
        else:
            raw_output = self._onnx_infer(blob)

        return self._postprocess(raw_output, orig_shape, scale, pad_top, pad_left)

    def _trt_infer(self, blob: np.ndarray) -> np.ndarray:
        np.copyto(self._host_inputs[0], blob.ravel())
        self._cuda.memcpy_htod_async(
            self._cuda_inputs[0], self._host_inputs[0], self._stream
        )
        self._context.execute_async_v2(
            bindings=self._bindings, stream_handle=self._stream.handle
        )
        self._cuda.memcpy_dtoh_async(
            self._host_outputs[0], self._cuda_outputs[0], self._stream
        )
        self._stream.synchronize()
        # Reshape to [1, 4+nc, 8400]
        output_shape = self._engine.get_binding_shape(
            self._engine.num_bindings - 1
        )
        return self._host_outputs[0].reshape(output_shape)[np.newaxis]

    def _onnx_infer(self, blob: np.ndarray) -> np.ndarray:
        outputs = self._ort_session.run(
            None, {self._ort_input_name: blob}
        )
        return outputs[0]

    # ── Performance benchmark ─────────────────────────────
    def benchmark(self, n: int = 100) -> Dict:
        """Measure mean latency and FPS over n random frames."""
        dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Warmup
        for _ in range(10):
            self.predict(dummy)

        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            self.predict(dummy)
            times.append(time.perf_counter() - t0)

        mean_ms = np.mean(times) * 1000
        fps     = 1.0 / np.mean(times)
        logger.info(
            f"Benchmark ({n} frames): "
            f"mean={mean_ms:.1f}ms  fps={fps:.1f}  "
            f"(target: <30ms, >30fps)"
        )
        return {"mean_ms": mean_ms, "fps": fps}
