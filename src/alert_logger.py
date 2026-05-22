#!/usr/bin/env python3
"""
alert_logger.py — PPE Violation Alert & Audit Logging Module
==============================================================
Handles three alert channels:
  1. On-screen overlay  (drawn by the caller using returned data)
  2. Audible buzzer     (WAV file via playsound / ALSA)
  3. HTTP webhook       (POST JSON to a remote endpoint)

And structured event logging:
  • Rotating log file (JSON Lines format)
  • JPEG frame snapshot saved to disk
  • Zone-aware — each event records the zone identifier
==============================================================
"""

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
from loguru import logger


# ─────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────
@dataclass
class Detection:
    """A single PPE detection result within a frame."""
    class_id:   int
    class_name: str
    confidence: float
    bbox_xyxy:  Tuple[int, int, int, int]   # (x1, y1, x2, y2) pixels


@dataclass
class ViolationEvent:
    """
    Structured record of a PPE violation for audit logging.

    Fields
    ──────
    timestamp       ISO-8601 UTC string
    zone_id         Zone identifier from deploy.yaml
    zone_label      Human-readable zone name
    missing_ppe     List of PPE item names that are absent
    detections      All detections present in the frame
    snapshot_path   Absolute path to the saved JPEG snapshot
    frame_index     Frame counter at time of violation
    """
    timestamp:     str
    zone_id:       str
    zone_label:    str
    missing_ppe:   List[str]
    detections:    List[Detection]
    snapshot_path: str
    frame_index:   int
    extra:         Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────
# Helper — Zone Hit Test
# ─────────────────────────────────────────────────────────
def detection_in_zone(
    bbox_xyxy: Tuple[int, int, int, int],
    zone_coords: Tuple[int, int, int, int],
) -> bool:
    """
    Return True if the bounding-box centroid falls inside the zone
    rectangle.  Both arguments are (x1, y1, x2, y2) pixel coords.
    """
    bx1, by1, bx2, by2 = bbox_xyxy
    zx1, zy1, zx2, zy2 = zone_coords

    cx = (bx1 + bx2) / 2
    cy = (by1 + by2) / 2

    return zx1 <= cx <= zx2 and zy1 <= cy <= zy2


# ─────────────────────────────────────────────────────────
# Violation Detector
# ─────────────────────────────────────────────────────────
REQUIRED_PPE = {"helmet", "vest", "gloves"}


def check_violations(detections: List[Detection]) -> List[str]:
    """
    Given a list of detections in the current frame, return a list of
    PPE class names that are MISSING (i.e. not detected above threshold).

    A worker is non-compliant if any required PPE item is absent.
    """
    detected_classes = {d.class_name.lower() for d in detections}
    missing = sorted(REQUIRED_PPE - detected_classes)
    return missing


# ─────────────────────────────────────────────────────────
# Alert Manager
# ─────────────────────────────────────────────────────────
class AlertManager:
    """
    Centralises all alert channels (overlay, buzzer, webhook)
    and enforces per-channel cooldowns to prevent alert floods.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Parameters
        ──────────
        config : dict loaded from deploy.yaml → alerts section
        """
        self._cfg = config
        self._last_buzzer_time: float = 0.0
        self._buzzer_lock = threading.Lock()

        # Pre-load sound file path
        self._sound_file = config.get("buzzer", {}).get(
            "sound_file", "assets/alert.wav"
        )
        self._buzzer_cooldown = config.get("buzzer", {}).get(
            "cooldown_seconds", 5
        )

        # Webhook settings
        self._webhook_url = config.get("webhook", {}).get("url", "")
        self._webhook_timeout = config.get("webhook", {}).get(
            "timeout_seconds", 2
        )

    # ── 1. On-screen overlay data ──────────────────────────
    def get_overlay_data(
        self,
        missing_ppe: List[str],
        zone_label:  str,
    ) -> Dict[str, Any]:
        """
        Returns a dict describing the overlay to be drawn on the frame
        by the caller (inference_gstreamer.py draws it with OpenCV).
        """
        if not missing_ppe:
            return {"type": "ok", "text": "PPE OK", "color": (0, 200, 0)}

        return {
            "type":  "violation",
            "text":  f"⚠ VIOLATION [{zone_label}]: Missing {', '.join(missing_ppe).upper()}",
            "color": (0, 0, 255),          # red in BGR
            "missing": missing_ppe,
        }

    # ── 2. Audible buzzer ─────────────────────────────────
    def trigger_buzzer(self) -> None:
        """
        Play alert.wav asynchronously.  Respects cooldown_seconds to
        avoid continuous buzzing.
        """
        if not self._cfg.get("buzzer", {}).get("enabled", False):
            return

        now = time.time()
        with self._buzzer_lock:
            if now - self._last_buzzer_time < self._buzzer_cooldown:
                return  # still in cooldown
            self._last_buzzer_time = now

        def _play():
            try:
                from playsound import playsound
                playsound(self._sound_file, block=True)
            except Exception as exc:
                # Fallback: system beep via ALSA
                logger.warning(f"playsound failed ({exc}); trying ALSA beep")
                os.system("aplay -q /usr/share/sounds/alsa/Front_Center.wav 2>/dev/null")

        threading.Thread(target=_play, daemon=True).start()
        logger.debug("Buzzer triggered")

    # ── 3. HTTP webhook ────────────────────────────────────
    def send_webhook(self, event: ViolationEvent) -> None:
        """
        POST a JSON payload to the configured webhook URL.
        Runs in a daemon thread so it never blocks the video pipeline.
        """
        if not self._cfg.get("webhook", {}).get("enabled", False):
            return

        payload = {
            "timestamp":   event.timestamp,
            "zone_id":     event.zone_id,
            "zone_label":  event.zone_label,
            "missing_ppe": event.missing_ppe,
            "snapshot":    event.snapshot_path,
            "frame_index": event.frame_index,
        }

        def _post():
            try:
                resp = requests.post(
                    self._webhook_url,
                    json=payload,
                    timeout=self._webhook_timeout,
                )
                if resp.status_code == 200:
                    logger.debug(f"Webhook delivered (HTTP 200)")
                else:
                    logger.warning(
                        f"Webhook returned HTTP {resp.status_code}"
                    )
            except requests.exceptions.RequestException as exc:
                logger.warning(f"Webhook failed: {exc}")

        threading.Thread(target=_post, daemon=True).start()

    # ── Unified trigger (all channels) ────────────────────
    def trigger_all(self, event: ViolationEvent) -> None:
        """Trigger buzzer + webhook simultaneously."""
        self.trigger_buzzer()
        self.send_webhook(event)


# ─────────────────────────────────────────────────────────
# Audit Logger
# ─────────────────────────────────────────────────────────
class AuditLogger:
    """
    Writes structured violation events to a rotating JSON-Lines log
    and saves JPEG frame snapshots with embedded metadata.
    """

    def __init__(self, log_dir: str, snapshot_dir: str,
                 jpeg_quality: int = 85):
        self._log_dir      = Path(log_dir)
        self._snapshot_dir = Path(snapshot_dir)
        self._jpeg_quality = jpeg_quality

        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)

        # One log file per day
        self._current_log_path: Optional[Path] = None
        self._log_file = None
        self._log_lock = threading.Lock()
        self._open_log_file()

    def _open_log_file(self) -> None:
        """Open (or rotate to) today's log file."""
        day_str = datetime.utcnow().strftime("%Y-%m-%d")
        new_path = self._log_dir / f"ppe_violations_{day_str}.jsonl"

        if new_path != self._current_log_path:
            if self._log_file:
                self._log_file.close()
            self._current_log_path = new_path
            self._log_file = open(new_path, "a", encoding="utf-8")
            logger.info(f"Audit log: {new_path}")

    # ── Save snapshot ─────────────────────────────────────
    def save_snapshot(
        self,
        frame: np.ndarray,
        timestamp_str: str,
        zone_id: str,
        frame_index: int,
    ) -> str:
        """
        Save a JPEG snapshot of the current frame.

        Filename format:
            {zone_id}_{YYYYMMDD_HHMMSS_ffffff}_{frame_index}.jpg

        Returns the absolute path of the saved file.
        """
        # Sanitise timestamp for filename
        ts_safe = timestamp_str.replace(":", "-").replace(".", "_")
        filename = f"{zone_id}_{ts_safe}_{frame_index:08d}.jpg"
        path = self._snapshot_dir / filename

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
        success = cv2.imwrite(str(path), frame, encode_params)

        if not success:
            logger.error(f"Failed to save snapshot: {path}")
            return ""

        logger.debug(f"Snapshot saved: {path}")
        return str(path.resolve())

    # ── Log event ─────────────────────────────────────────
    def log_event(self, event: ViolationEvent) -> None:
        """
        Append a JSON-Lines record to the daily log file.

        Each line is a self-contained JSON object — easy to ingest
        with jq, Pandas, Splunk, or any SIEM.
        """
        record = {
            "timestamp":     event.timestamp,
            "zone_id":       event.zone_id,
            "zone_label":    event.zone_label,
            "missing_ppe":   event.missing_ppe,
            "frame_index":   event.frame_index,
            "snapshot_path": event.snapshot_path,
            "detections": [
                {
                    "class_id":   d.class_id,
                    "class_name": d.class_name,
                    "confidence": round(d.confidence, 4),
                    "bbox":       list(d.bbox_xyxy),
                }
                for d in event.detections
            ],
            **event.extra,
        }

        with self._log_lock:
            self._open_log_file()   # rotate if day changed
            self._log_file.write(json.dumps(record) + "\n")
            self._log_file.flush()

        logger.info(
            f"[VIOLATION] zone={event.zone_id} "
            f"missing={event.missing_ppe} "
            f"frame={event.frame_index}"
        )

    def close(self):
        if self._log_file:
            self._log_file.close()


# ─────────────────────────────────────────────────────────
# Convenience — create a ViolationEvent
# ─────────────────────────────────────────────────────────
def make_violation_event(
    frame:       np.ndarray,
    detections:  List[Detection],
    missing_ppe: List[str],
    zone_id:     str,
    zone_label:  str,
    frame_index: int,
    audit_logger: "AuditLogger",
) -> ViolationEvent:
    """
    Build a ViolationEvent, saving a snapshot automatically.
    """
    ts = datetime.utcnow().isoformat(timespec="microseconds") + "Z"

    snapshot_path = audit_logger.save_snapshot(
        frame=frame,
        timestamp_str=ts,
        zone_id=zone_id,
        frame_index=frame_index,
    )

    return ViolationEvent(
        timestamp=ts,
        zone_id=zone_id,
        zone_label=zone_label,
        missing_ppe=missing_ppe,
        detections=detections,
        snapshot_path=snapshot_path,
        frame_index=frame_index,
    )
