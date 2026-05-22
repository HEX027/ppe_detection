#!/usr/bin/env python3
"""
export_tensorrt.py — Export YOLOv8 → ONNX → TensorRT Engine
==============================================================
Usage:
    # Step 1: Export to ONNX
    python3 src/export_tensorrt.py --mode onnx \
        --weights models/weights/best.pt \
        --output  models/onnx/ppe_model.onnx

    # Step 2: Convert ONNX → TensorRT (FP16)
    python3 src/export_tensorrt.py --mode tensorrt \
        --onnx    models/onnx/ppe_model.onnx \
        --output  models/tensorrt/ppe_fp16.engine \
        --precision fp16

    # Step 3 (Optional): INT8 with calibration
    python3 src/export_tensorrt.py --mode tensorrt \
        --onnx    models/onnx/ppe_model.onnx \
        --output  models/tensorrt/ppe_int8.engine \
        --precision int8 \
        --calib-data data/processed/images/val

    # All-in-one (export + convert)
    python3 src/export_tensorrt.py --mode all \
        --weights models/weights/best.pt \
        --output  models/tensorrt/ppe_fp16.engine \
        --precision fp16

NOTES:
  • Requires TensorRT ≥ 8.5 (JetPack 5.x ships TRT 8.5+)
  • trtexec is an alternative CLI shipped with TensorRT — see
    the bash commands printed at the end of this script.
==============================================================
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export YOLOv8 weights to ONNX and/or TensorRT"
    )
    parser.add_argument(
        "--mode", choices=["onnx", "tensorrt", "all"],
        default="all", help="Export step to perform"
    )
    parser.add_argument(
        "--weights", type=str,
        default="models/weights/best.pt",
        help="Path to YOLOv8 .pt checkpoint"
    )
    parser.add_argument(
        "--onnx", type=str,
        default="models/onnx/ppe_model.onnx",
        help="Path to input/output ONNX file"
    )
    parser.add_argument(
        "--output", type=str,
        default="models/tensorrt/ppe_fp16.engine",
        help="Path for the TensorRT engine output"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Square input resolution"
    )
    parser.add_argument(
        "--batch", type=int, default=1,
        help="Max batch size for TensorRT engine"
    )
    parser.add_argument(
        "--precision", choices=["fp32", "fp16", "int8"],
        default="fp16", help="TensorRT precision mode"
    )
    parser.add_argument(
        "--calib-data", type=str,
        default="data/processed/images/val",
        help="Directory of calibration images for INT8"
    )
    parser.add_argument(
        "--opset", type=int, default=17,
        help="ONNX opset version"
    )
    parser.add_argument(
        "--simplify", action="store_true", default=True,
        help="Run onnx-simplifier before TRT conversion"
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────
# Step 1 — YOLOv8 → ONNX
# ─────────────────────────────────────────────────────────
def export_to_onnx(
    weights_path: str,
    onnx_path: str,
    imgsz: int = 640,
    opset: int = 17,
    simplify: bool = True,
) -> str:
    """
    Export a trained YOLOv8 .pt model to ONNX format using
    the Ultralytics built-in exporter.

    The exporter:
      • Traces the model with a dummy input [1, 3, imgsz, imgsz]
      • Embeds NMS as a post-processing node (dynamic_axes)
      • Optionally simplifies the graph with onnxsim
    """
    from ultralytics import YOLO

    logger.info(f"Exporting {weights_path} → ONNX …")
    model = YOLO(weights_path)

    # Ultralytics export() writes to <weights_dir>/<name>.onnx
    export_result = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=simplify,
        dynamic=False,    # static batch for TRT (batch=1 on Jetson)
        half=False,       # export in FP32; quantise at TRT step
        int8=False,
        device="0",
    )

    # Ultralytics returns the output path as a string
    generated_path = str(export_result)
    logger.info(f"ONNX written to: {generated_path}")

    # Move to canonical output path if different
    if Path(generated_path).resolve() != Path(onnx_path).resolve():
        import shutil
        Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(generated_path, onnx_path)
        logger.success(f"ONNX moved to: {onnx_path}")

    # ── Verify ONNX graph ──────────────────────────────────
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    logger.success("ONNX graph check passed ✅")

    # Print input/output tensor shapes
    for inp in onnx_model.graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        logger.info(f"  Input  : {inp.name}  shape={shape}")
    for out in onnx_model.graph.output:
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        logger.info(f"  Output : {out.name}  shape={shape}")

    return onnx_path


# ─────────────────────────────────────────────────────────
# INT8 Calibration Dataset
# ─────────────────────────────────────────────────────────
class Int8CalibrationDataset:
    """
    Provides calibration batches for TensorRT INT8 quantisation.
    Loads JPEG/PNG images from a directory, resizes to model input
    size, and normalises to [0, 1].
    """

    def __init__(self, image_dir: str, imgsz: int = 640, n_batches: int = 50):
        import glob
        self.imgsz = imgsz
        self.n_batches = n_batches
        self.image_paths = sorted(
            glob.glob(os.path.join(image_dir, "*.jpg")) +
            glob.glob(os.path.join(image_dir, "*.png"))
        )
        if not self.image_paths:
            raise FileNotFoundError(
                f"No calibration images found in {image_dir}"
            )
        logger.info(
            f"INT8 calibration: {len(self.image_paths)} images "
            f"in {image_dir}"
        )

    def __len__(self):
        return min(self.n_batches, len(self.image_paths))

    def __iter__(self):
        import cv2
        for path in self.image_paths[: len(self)]:
            img = cv2.imread(path)
            img = cv2.resize(img, (self.imgsz, self.imgsz))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Normalise to [0, 1], shape [1, 3, H, W], float32
            tensor = (
                img.astype(np.float32) / 255.0
            ).transpose(2, 0, 1)[np.newaxis, ...]
            yield tensor


# ─────────────────────────────────────────────────────────
# Step 2 — ONNX → TensorRT Engine (Python API)
# ─────────────────────────────────────────────────────────
def build_tensorrt_engine(
    onnx_path: str,
    engine_path: str,
    imgsz: int = 640,
    max_batch: int = 1,
    precision: str = "fp16",
    calib_data_dir: str = "",
) -> str:
    """
    Build a TensorRT engine from an ONNX model using the
    TensorRT Python API (tensorrt 8.x).

    Precision modes:
      fp32  — full float32 (slowest, highest accuracy)
      fp16  — half precision (recommended for Jetson)
      int8  — 8-bit integer (fastest, needs calibration data)
    """
    try:
        import tensorrt as trt
    except ImportError:
        logger.error(
            "tensorrt Python bindings not found.\n"
            "On Jetson, TensorRT is provided by JetPack.\n"
            "Alternatively, use the trtexec CLI (see printed commands below)."
        )
        _print_trtexec_commands(onnx_path, engine_path, precision)
        sys.exit(1)

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")

    logger.info(f"Building TensorRT engine [{precision.upper()}] …")
    logger.info(f"  ONNX  : {onnx_path}")
    logger.info(f"  Engine: {engine_path}")

    Path(engine_path).parent.mkdir(parents=True, exist_ok=True)

    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser  = trt.OnnxParser(network, TRT_LOGGER)
    config  = builder.create_builder_config()

    # ── Memory & precision ────────────────────────────────
    # 2 GB workspace on Jetson Orin Nano (8 GB RAM variant)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            logger.warning("FP16 not supported on this platform; falling back to FP32")
        else:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("FP16 mode enabled")

    elif precision == "int8":
        if not builder.platform_has_fast_int8:
            logger.warning("INT8 not supported; falling back to FP16")
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            config.set_flag(trt.BuilderFlag.INT8)
            logger.info("INT8 mode enabled — loading calibration data …")

            # INT8 entropy calibrator
            calib_dataset = Int8CalibrationDataset(
                image_dir=calib_data_dir,
                imgsz=imgsz,
            )

            class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
                def __init__(self, dataset, cache_file="int8_calib.cache"):
                    super().__init__()
                    self._dataset   = iter(dataset)
                    self._cache     = cache_file
                    self._n_batches = len(dataset)
                    self._idx       = 0
                    import pycuda.driver as cuda
                    import pycuda.autoinit  # noqa: F401
                    self._cuda = cuda
                    sample = next(iter(Int8CalibrationDataset(
                        calib_data_dir, imgsz, n_batches=1
                    )))
                    self._buf = cuda.mem_alloc(sample.nbytes)

                def get_batch_size(self):
                    return 1

                def get_batch(self, names):
                    if self._idx >= self._n_batches:
                        return None
                    try:
                        batch = next(self._dataset)
                        self._cuda.memcpy_htod(self._buf, batch)
                        self._idx += 1
                        return [int(self._buf)]
                    except StopIteration:
                        return None

                def read_calibration_cache(self):
                    if os.path.exists(self._cache):
                        with open(self._cache, "rb") as f:
                            return f.read()
                    return None

                def write_calibration_cache(self, cache):
                    with open(self._cache, "wb") as f:
                        f.write(cache)

            config.int8_calibrator = EntropyCalibrator(calib_dataset)

    # ── Parse ONNX ────────────────────────────────────────
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error(f"ONNX parse error: {parser.get_error(i)}")
            sys.exit(1)
    logger.info("ONNX parsed successfully")

    # ── Optimisation profile (static batch=1) ─────────────
    profile = builder.create_optimization_profile()
    input_name  = network.get_input(0).name
    input_shape = (max_batch, 3, imgsz, imgsz)
    profile.set_shape(input_name, input_shape, input_shape, input_shape)
    config.add_optimization_profile(profile)

    # ── Build serialised engine ───────────────────────────
    t0 = time.time()
    logger.info("Building engine — this may take 5-15 minutes on first run …")
    serialised_engine = builder.build_serialized_network(network, config)
    if serialised_engine is None:
        logger.error("Engine build FAILED")
        sys.exit(1)

    with open(engine_path, "wb") as f:
        f.write(serialised_engine)

    elapsed = time.time() - t0
    size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    logger.success(
        f"TensorRT engine saved: {engine_path} "
        f"({size_mb:.1f} MB, built in {elapsed:.0f}s)"
    )
    return engine_path


# ─────────────────────────────────────────────────────────
# trtexec CLI Commands (alternative to Python API)
# ─────────────────────────────────────────────────────────
def _print_trtexec_commands(
    onnx_path: str,
    engine_path: str,
    precision: str,
) -> None:
    """Print equivalent trtexec shell commands for reference."""
    fp16_flag = "--fp16" if precision in ("fp16", "int8") else ""
    int8_flag = "--int8" if precision == "int8" else ""

    logger.info("\n" + "=" * 60)
    logger.info("  Equivalent trtexec commands (run on Jetson):")
    logger.info("=" * 60)

    # FP16
    logger.info("""
# ── FP16 Engine ──────────────────────────────────────────
trtexec \\
    --onnx={onnx} \\
    --saveEngine={engine_fp16} \\
    --fp16 \\
    --workspace=2048 \\
    --verbose
""".format(
        onnx=onnx_path,
        engine_fp16=engine_path.replace(".engine", "_fp16.engine"),
    ))

    # INT8
    logger.info("""
# ── INT8 Engine (with calibration) ───────────────────────
trtexec \\
    --onnx={onnx} \\
    --saveEngine={engine_int8} \\
    --int8 \\
    --fp16 \\
    --calib=models/tensorrt/int8_calib.cache \\
    --workspace=2048 \\
    --verbose
""".format(
        onnx=onnx_path,
        engine_int8=engine_path.replace(".engine", "_int8.engine"),
    ))

    # Benchmark
    logger.info("""
# ── Latency / throughput benchmark ───────────────────────
trtexec \\
    --loadEngine={engine} \\
    --batch=1 \\
    --iterations=100 \\
    --avgRuns=10
""".format(engine=engine_path))
    logger.info("=" * 60)


# ─────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    onnx_output = args.onnx

    if args.mode in ("onnx", "all"):
        onnx_output = export_to_onnx(
            weights_path=args.weights,
            onnx_path=args.onnx,
            imgsz=args.imgsz,
            opset=args.opset,
            simplify=args.simplify,
        )

    if args.mode in ("tensorrt", "all"):
        build_tensorrt_engine(
            onnx_path=onnx_output,
            engine_path=args.output,
            imgsz=args.imgsz,
            max_batch=args.batch,
            precision=args.precision,
            calib_data_dir=args.calib_data,
        )
        # Print CLI alternative for reference
        _print_trtexec_commands(onnx_output, args.output, args.precision)
