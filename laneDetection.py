"""Run the official hustvl/YOLOP ONNX model on a MOV/MP4 video.

No training or video-specific calibration. Red = lanes; green = drivable area.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time
import urllib.request

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent
REVISION = "8d8f68df318c71f01d6f813c024df646c7d1978f"
MODEL_URL = f"https://raw.githubusercontent.com/hustvl/YOLOP/{REVISION}/weights/yolop-640-640.onnx"
DEFAULT_MODEL = ROOT / "weights" / "yolop-640-640.onnx"


def get_model(path: Path) -> Path:
    if path.is_file():
        return path
    if path.resolve() != DEFAULT_MODEL.resolve():
        raise FileNotFoundError(f"Custom model does not exist: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading official YOLOP weights (about 36 MB)...", flush=True)
    temporary = None
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=120) as response:
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as stream:
                temporary = Path(stream.name)
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
        if temporary.stat().st_size != 35902568:
            raise RuntimeError("Unexpected model download size; try again.")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def preprocess(frame: np.ndarray, height: int, width: int):
    """Official RGB + ImageNet normalization, with aspect-preserving padding."""
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    rw, rh = max(1, round(w * scale)), max(1, round(h * scale))
    left, top = (width - rw) // 2, (height - rh) // 2
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (rw, rh), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 114, dtype=np.uint8)
    canvas[top:top + rh, left:left + rw] = rgb
    tensor = canvas.astype(np.float32) / 255.0
    tensor = (tensor - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
    return np.ascontiguousarray(tensor.transpose(2, 0, 1)[None]), (left, top, rw, rh)


def restore_mask(scores, padding, shape):
    """Remove letterbox padding before restoring a binary mask to video size."""
    left, top, rw, rh = padding
    if scores.ndim != 4 or scores.shape[:2] != (1, 2):
        raise ValueError(f"Expected segmentation [1, 2, H, W], got {scores.shape}")
    cropped = scores[0, :, top:top + rh, left:left + rw]
    mask = np.argmax(cropped, axis=0).astype(np.uint8)
    return cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


class YOLOP:
    def __init__(self, model=DEFAULT_MODEL, device="cpu"):
        model = get_model(Path(model))
        available = ort.get_available_providers()
        if device == "cuda" and "CUDAExecutionProvider" not in available:
            raise RuntimeError("CUDA provider unavailable. Install compatible onnxruntime-gpu/CUDA, or use --device cpu.")
        providers = ["CPUExecutionProvider"] if device == "cpu" else ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model), providers=providers)
        if device == "cuda" and "CUDAExecutionProvider" not in self.session.get_providers():
            raise RuntimeError("CUDA initialization failed. Check CUDA/cuDNN or use --device cpu.")
        inputs = self.session.get_inputs()
        if len(inputs) != 1 or inputs[0].type != "tensor(float)":
            raise ValueError("Expected the official float32 YOLOP ONNX model.")
        self.input_name = inputs[0].name
        shape = inputs[0].shape
        if len(shape) != 4 or shape[:2] != [1, 3] or not all(isinstance(x, int) and x > 0 for x in shape[2:]):
            raise ValueError(f"Expected fixed [1, 3, H, W] model input, got {shape}")
        self.height, self.width = shape[2:]
        names = {output.name for output in self.session.get_outputs()}
        if not {"lane_line_seg", "drive_area_seg"} <= names:
            raise ValueError(f"YOLOP segmentation outputs missing: {names}")
        self.model_path = model

    def predict(self, frame, include_road=False):
        tensor, padding = preprocess(frame, self.height, self.width)
        names = ["lane_line_seg"] + (["drive_area_seg"] if include_road else [])
        outputs = self.session.run(names, {self.input_name: tensor})
        if any(x.shape[2:] != (self.height, self.width) for x in outputs):
            raise ValueError("Segmentation dimensions do not match model input.")
        lane = restore_mask(outputs[0], padding, frame.shape)
        road = restore_mask(outputs[1], padding, frame.shape) if include_road else None
        return lane, road


def overlay(frame, lane, road=None, alpha=.6):
    color = frame.copy()
    if road is not None:
        color[road == 1] = (0, 200, 0)
    color[lane == 1] = (0, 0, 255)
    return cv2.addWeighted(frame, 1 - alpha, color, alpha, 0)


def run(args):
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.output.suffix.lower() != ".mp4":
        raise ValueError("Use .mp4 for the output file.")
    report_path = args.output.with_suffix(".json")
    if args.output.exists() or report_path.exists():
        raise FileExistsError("Output or report already exists; choose a new output name.")
    if args.mask_dir and args.mask_dir.exists() and any(args.mask_dir.iterdir()):
        raise FileExistsError("Mask directory must be empty or new.")
    if args.max_frames < 0 or not 0 <= args.alpha <= 1:
        raise ValueError("max-frames must be >= 0; alpha must be between 0 and 1.")
    model = YOLOP(args.model, args.device)
    cap = cv2.VideoCapture(str(args.input))
    writer = None
    count = 0
    inference_seconds = 0.0
    stopped = False
    start = time.perf_counter()
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {args.input}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise RuntimeError("Cannot read input FPS; convert the input to a valid constant-FPS video first.")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.mask_dir:
            args.mask_dir.mkdir(parents=True, exist_ok=True)
        original_shape = None
        while not args.max_frames or count < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if original_shape is None:
                original_shape = frame.shape[:2]
            elif original_shape != frame.shape[:2]:
                raise RuntimeError("Video frame dimensions changed during decoding.")
            before = time.perf_counter()
            lane, road = model.predict(frame, args.show_road)
            inference_seconds += time.perf_counter() - before
            rendered = overlay(frame, lane, road, args.alpha)
            # MP4 requires even dimensions: pad at most one row/column, never crop.
            h, w = rendered.shape[:2]
            rendered = cv2.copyMakeBorder(rendered, 0, h % 2, 0, w % 2, cv2.BORDER_REPLICATE)
            if writer is None:
                writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                         (rendered.shape[1], rendered.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError("Cannot create MP4 video writer.")
            writer.write(rendered)
            if args.mask_dir:
                success, encoded = cv2.imencode(".png", lane * 255)
                if not success:
                    raise RuntimeError("Cannot encode lane mask.")
                encoded.tofile(str(args.mask_dir / f"{count:06d}.png"))
            count += 1
            if count % 30 == 0:
                print(f"Processed {count}/{total if total > 0 else '?'} frames", flush=True)
            if args.preview:
                ratio = min(1., 960 / rendered.shape[1], 800 / rendered.shape[0])
                cv2.imshow("YOLOP | red: lanes, green: road | Q: stop", cv2.resize(rendered, None, fx=ratio, fy=ratio))
                if cv2.waitKey(1) & 0xff == ord("q"):
                    stopped = True
                    break
        if count == 0:
            raise RuntimeError("No video frames decoded.")
        if total > 0 and count < total and not stopped and (not args.max_frames or count < args.max_frames):
            raise RuntimeError(f"Decode stopped early ({count}/{total}). Output is incomplete.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.preview:
            cv2.destroyAllWindows()
    report = {
        "input": str(args.input.resolve()), "output": str(args.output.resolve()),
        "model": str(model.model_path.resolve()),
        "model_sha256": hashlib.sha256(model.model_path.read_bytes()).hexdigest(),
        "official_download_revision": REVISION if model.model_path.resolve() == DEFAULT_MODEL.resolve() else None,
        "providers": model.session.get_providers(), "frames": count, "source_fps": fps,
        "original_size_hw": original_shape, "elapsed_seconds": time.perf_counter() - start,
        "mean_inference_ms": inference_seconds / count * 1000,
        "stopped_by_user": stopped,
        "note": "Visual inference only, not accuracy evaluation. No audio. Constant-FPS output."
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {count} frames: {args.output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Official YOLOP pretrained model: MOV/MP4 lane segmentation")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("yolop_result.mp4"))
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Official ONNX model; auto-download default if missing")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--show-road", action="store_true", help="Also overlay the drivable area")
    parser.add_argument("--mask-dir", type=Path, help="Save original-size binary lane PNGs (0/255)")
    parser.add_argument("--max-frames", type=int, default=0, help="0: entire video")
    parser.add_argument("--alpha", type=float, default=.6)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
