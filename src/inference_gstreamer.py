#!/usr/bin/env python3
"""
inference_gstreamer.py — Real-Time PPE Detection Inference Pipeline
====================================================================
Entry point for the deployed system on the NVIDIA Jetson Orin Nano.

Pipeline overview
─────────────────
GStreamer source → appsink → TensorRT inference → overlay → display
                                     ↓
                             Alert Manager (buzzer, webhook)
                                     ↓
                              Audit Logger (JSONL + JPEG)

Supported sources (set in configs/deploy.yaml → source.uri):
  • /dev/video0          — USB / CSI camera
  • rtsp://...           — IP camera (H.264/H.265)
  • /path/to/video.mp4   — offline video file (testing)

Usage
─────
    python3 src/inference_gstreamer.py --config configs/deploy.yaml
    python3 src/inference_gstreamer.py --config configs/deploy.yaml \
                                       --source /dev/video0 \
                                       --show-fps
====================================================================
"""

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.alert_logger import (
    AlertManager,
    AuditLogger,
    Detection,
    ViolationEvent,
    check_violations,
    detection_in_zone,
    make_violation_event,
)
from src.tensorrt_infer import TRTInferenceEngine


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PPE Real-Time Detection — GStreamer / OpenCV Pipeline"
    )
    parser.add_argument(
        "--config", type=str,
        default="configs/deploy.yaml",
        help="Path to deploy.yaml"
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Override source URI from config"
    )
    parser.add_argument(
        "--show-fps", action="store_true",
        help="Display FPS counter on overlay"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Disable on-screen window (headless mode)"
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run inference benchmark then exit"
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────
# Config Loader
# ─────────────────────────────────────────────────────────
def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Config loaded from: {path}")
    return cfg


# ─────────────────────────────────────────────────────────
# GStreamer Pipeline Builder
# ─────────────────────────────────────────────────────────
def build_gst_pipeline(source_uri: str, width: int, height: int, fps: int) -> str:
    """
    Return a GStreamer pipeline string appropriate for the source.

    The pipeline ends with:
      videoconvert → video/x-raw,format=BGR → appsink
    so that OpenCV VideoCapture can consume it directly.

    For Jetson we use nvv4l2decoder for H.264/H.265 hardware decoding
    and nvvidconv for fast colour space conversion on the VIC engine.
    """
    appsink = (
        "appsink max-buffers=1 drop=true sync=false "
        'caps="video/x-raw,format=BGR"'
    )

    # ── Camera (V4L2) ─────────────────────────────────────
    if source_uri.startswith("/dev/video"):
        pipeline = (
            f"v4l2src device={source_uri} ! "
            f"video/x-raw,width={width},height={height},framerate={fps}/1 ! "
            f"nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! "
            f"nvvidconv ! video/x-raw,format=BGRx ! "
            f"videoconvert ! video/x-raw,format=BGR ! "
            f"{appsink}"
        )

    # ── RTSP stream ───────────────────────────────────────
    elif source_uri.startswith("rtsp://"):
        pipeline = (
            f"rtspsrc location={source_uri} latency=0 ! "
            f"rtph264depay ! h264parse ! nvv4l2decoder ! "
            f"nvvidconv ! video/x-raw,format=BGRx ! "
            f"videoconvert ! video/x-raw,format=BGR ! "
            f"videoscale ! video/x-raw,width={width},height={height} ! "
            f"{appsink}"
        )

    # ── File / test source ────────────────────────────────
    else:
        pipeline = (
            f"filesrc location={source_uri} ! decodebin ! "
            f"videoconvert ! videoscale ! "
            f"video/x-raw,format=BGR,width={width},height={height} ! "
            f"{appsink}"
        )

    logger.debug(f"GStreamer pipeline:\n  {pipeline}")
    return pipeline


# ─────────────────────────────────────────────────────────
# Overlay Renderer
# ─────────────────────────────────────────────────────────
class OverlayRenderer:
    """Draws bounding boxes, class labels, zone outlines, and alert banners."""

    # BGR colours for each class
    CLASS_COLORS = {
        "helmet": (0, 255,   0),    # green
        "vest":   (0, 165, 255),    # orange
        "gloves": (255,  0,   0),   # blue
    }
    VIOLATION_COLOR = (0, 0, 255)   # red
    OK_COLOR        = (0, 200,  0)  # green

    def draw(
        self,
        frame:       np.ndarray,
        detections:  List[Detection],
        zones:       List[Dict],
        missing_ppe: List[str],
        fps:         float,
        frame_index: int,
        show_fps:    bool = True,
    ) -> np.ndarray:
        """
        Annotate frame in-place and return it.

        Draws:
          • Zone rectangles (blue dashed outline)
          • Bounding boxes per detection (class-specific colour)
          • Confidence label above each box
          • Alert banner at top when PPE is missing
          • FPS counter (bottom-left)
        """
        overlay = frame.copy()

        # ── Zone outlines ──────────────────────────────────
        for zone in zones:
            zx1, zy1, zx2, zy2 = zone["coords"]
            cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2),
                          (255, 200, 0), 2, cv2.LINE_AA)
            cv2.putText(
                overlay, zone.get("label", zone["id"]),
                (zx1 + 6, zy1 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2
            )

        # ── Bounding boxes ────────────────────────────────
        for det in detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            color = self.CLASS_COLORS.get(det.class_name, (200, 200, 200))
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            label = f"{det.class_name} {det.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
            )
            cv2.rectangle(overlay, (x1, y1 - th - 8), (x1 + tw + 4, y1),
                          color, -1)
            cv2.putText(overlay, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

        # ── Alert banner ───────────────────────────────────
        h, w = frame.shape[:2]
        if missing_ppe:
            banner_color = self.VIOLATION_COLOR
            msg = f"  ⚠ PPE VIOLATION — Missing: {', '.join(missing_ppe).upper()}  "
        else:
            banner_color = self.OK_COLOR
            msg = "  ✓ PPE COMPLIANT  "

        cv2.rectangle(overlay, (0, 0), (w, 40), banner_color, -1)
        cv2.putText(overlay, msg, (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # ── FPS counter ────────────────────────────────────
        if show_fps:
            fps_text = f"FPS: {fps:.1f}  Frame: {frame_index}"
            cv2.putText(overlay, fps_text, (8, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Blend overlay with original frame (80% overlay)
        return cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)


# ─────────────────────────────────────────────────────────
# Main Inference Loop
# ─────────────────────────────────────────────────────────
class PPEDetectionPipeline:
    """
    Orchestrates the complete real-time pipeline:
      capture → infer → alert → log → display
    """

    def __init__(self, cfg: Dict[str, Any], show_fps: bool, no_display: bool):
        self._cfg        = cfg
        self._show_fps   = show_fps
        self._no_display = no_display
        self._running    = False

        # ── Load TensorRT engine ───────────────────────────
        model_cfg = cfg["model"]
        self._engine = TRTInferenceEngine(
            engine_path=model_cfg["engine_path"],
            class_names=model_cfg["class_names"],
            conf_thresh=model_cfg["conf_thresh"],
            iou_thresh=model_cfg["iou_thresh"],
            input_size=tuple(model_cfg["input_size"]),
        )

        # ── Zones ─────────────────────────────────────────
        self._zones: List[Dict] = cfg.get("zones", [])
        if not self._zones:
            # Default: full frame
            src = cfg["source"]
            self._zones = [{
                "id": "zone_default", "label": "Full Frame",
                "coords": [0, 0, src["width"], src["height"]],
            }]

        # ── Alert manager ──────────────────────────────────
        self._alert_mgr = AlertManager(cfg.get("alerts", {}))

        # ── Audit logger ───────────────────────────────────
        log_cfg = cfg.get("logging", {})
        self._audit = AuditLogger(
            log_dir=log_cfg.get("log_dir", "logs/events"),
            snapshot_dir=log_cfg.get("snapshot_dir", "logs/snapshots"),
            jpeg_quality=log_cfg.get("jpeg_quality", 85),
        )

        # ── Overlay renderer ───────────────────────────────
        self._renderer = OverlayRenderer()

        # ── FPS tracking ───────────────────────────────────
        self._fps_alpha = 0.05    # EMA smoothing factor
        self._fps_ema   = 30.0

    # ── Camera / GStreamer capture ─────────────────────────
    def _open_capture(self) -> cv2.VideoCapture:
        src = self._cfg["source"]
        pipeline = build_gst_pipeline(
            source_uri=src["uri"],
            width=src["width"],
            height=src["height"],
            fps=src["fps"],
        )

        # Try GStreamer backend first; fall back to default (file/webcam)
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            logger.warning(
                "GStreamer pipeline failed — falling back to cv2 default backend"
            )
            cap = cv2.VideoCapture(src["uri"])

        if not cap.isOpened():
            logger.error(f"Cannot open source: {src['uri']}")
            sys.exit(1)

        logger.success(
            f"Video source opened: {src['uri']}  "
            f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×"
            f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
        )
        return cap

    # ── Per-frame logic ────────────────────────────────────
    def _process_frame(
        self,
        frame:       np.ndarray,
        frame_index: int,
    ) -> Tuple[np.ndarray, bool]:
        """
        Run inference on a single frame.

        Returns
        ───────
        annotated_frame : BGR frame with overlay
        violation       : True if any PPE is missing
        """
        # ── Inference ─────────────────────────────────────
        detections = self._engine.predict(frame)

        # ── Zone analysis ─────────────────────────────────
        # For each zone, check which detections fall inside it
        # and whether required PPE is present.
        worst_missing: List[str] = []
        active_zone: Optional[Dict] = None

        for zone in self._zones:
            zone_dets = [
                d for d in detections
                if detection_in_zone(d.bbox_xyxy, tuple(zone["coords"]))
            ]
            missing = check_violations(zone_dets)

            if missing and len(missing) > len(worst_missing):
                worst_missing = missing
                active_zone   = zone

        # ── Alert & log if violation ───────────────────────
        violation = bool(worst_missing)
        if violation and active_zone is not None:
            event = make_violation_event(
                frame=frame,
                detections=detections,
                missing_ppe=worst_missing,
                zone_id=active_zone["id"],
                zone_label=active_zone.get("label", active_zone["id"]),
                frame_index=frame_index,
                audit_logger=self._audit,
            )
            self._audit.log_event(event)
            self._alert_mgr.trigger_all(event)

        # ── Draw overlay ───────────────────────────────────
        annotated = self._renderer.draw(
            frame=frame,
            detections=detections,
            zones=self._zones,
            missing_ppe=worst_missing,
            fps=self._fps_ema,
            frame_index=frame_index,
            show_fps=self._show_fps,
        )
        return annotated, violation

    # ── Main run loop ─────────────────────────────────────
    def run(self) -> None:
        cap = self._open_capture()

        self._running  = True
        frame_index    = 0
        skip_frames    = self._cfg.get("performance", {}).get("skip_frames", 0)
        warmup_frames  = self._cfg.get("performance", {}).get("warmup_frames", 10)
        t_prev         = time.perf_counter()

        logger.info("Pipeline running — press Ctrl+C or 'q' to stop")

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("End of stream or read error — stopping")
                    break

                frame_index += 1

                # ── Frame skip (optional) ──────────────────
                if skip_frames > 0 and (frame_index % (skip_frames + 1)) != 0:
                    continue

                # ── FPS measurement ────────────────────────
                t_now = time.perf_counter()
                inst_fps = 1.0 / max(t_now - t_prev, 1e-6)
                self._fps_ema = (
                    self._fps_alpha * inst_fps
                    + (1 - self._fps_alpha) * self._fps_ema
                )
                t_prev = t_now

                # Skip FPS measurement for warmup frames
                if frame_index <= warmup_frames:
                    continue

                # ── Process ────────────────────────────────
                annotated, violation = self._process_frame(frame, frame_index)

                # ── Display ────────────────────────────────
                if not self._no_display:
                    cv2.imshow("PPE Detection — Jetson Orin Nano", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:   # 'q' or ESC
                        logger.info("User quit")
                        break

                # ── Periodic stats ─────────────────────────
                if frame_index % 300 == 0:
                    logger.info(
                        f"Frame {frame_index:>8}  "
                        f"FPS={self._fps_ema:.1f}  "
                        f"Violation={'YES' if violation else 'no '}"
                    )

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt — shutting down …")
        finally:
            self._shutdown(cap)

    def _shutdown(self, cap: cv2.VideoCapture) -> None:
        self._running = False
        cap.release()
        cv2.destroyAllWindows()
        self._audit.close()
        logger.success("Pipeline stopped cleanly")

    def stop(self) -> None:
        self._running = False


# ─────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    # Configure loguru
    logger.remove()
    logger.add(sys.stderr, level="INFO", colorize=True)
    logger.add(
        "logs/events/pipeline_{time}.log",
        level="DEBUG",
        rotation="50 MB",
        compression="gz",
    )

    # Load config
    cfg = load_config(args.config)

    # Override source if CLI flag provided
    if args.source:
        cfg["source"]["uri"] = args.source

    # Build pipeline
    pipeline = PPEDetectionPipeline(
        cfg=cfg,
        show_fps=args.show_fps,
        no_display=args.no_display,
    )

    # Benchmark mode
    if args.benchmark:
        logger.info("Running inference benchmark …")
        pipeline._engine.benchmark(n=200)
        return

    # Register SIGTERM handler for clean Docker stop
    def _sigterm_handler(signum, frame):
        logger.info("SIGTERM received — stopping …")
        pipeline.stop()

    signal.signal(signal.SIGTERM, _sigterm_handler)

    # Run
    pipeline.run()


if __name__ == "__main__":
    main()
