#!/usr/bin/env python3
"""Local Lyra-2 demo UI.

This script intentionally avoids web-framework dependencies.  It serves a small
browser UI, writes Lyra-compatible trajectory/caption files, and launches the
existing inference modules in background jobs.
"""

from __future__ import annotations

import argparse
import cgi
import dataclasses
import datetime as dt
import html
import io
import json
import math
import mimetypes
import os
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency fallback
    Image = None

try:
    import cv2
except Exception:  # pragma: no cover - Lyra env normally provides OpenCV
    cv2 = None


ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "outputs" / "ui_runs"
DEFAULT_PROMPT = "a realistic explorable scene with consistent geometry and natural lighting"
SETTINGS_FILE = ROOT / ".ui_settings.json"
CHUNK_SIZE = 80
AR_SEQUENCE_VIDEO_NAME = "ar_sequence.mp4"

PRESET_TRAJECTORIES = {
    "forward": "Forward walk",
    "backward": "Backward retreat",
    "slide_left": "Slide left",
    "slide_right": "Slide right",
    "orbit_left": "Orbit left",
    "orbit_right": "Orbit right",
    "pan_left": "Look left",
    "pan_right": "Look right",
    "spiral_forward": "Spiral forward",
    "rotate_zoom": "Rotate and push in",
}

OP_LABELS = {
    "w": "W forward",
    "s": "S back",
    "a": "A left",
    "d": "D right",
    "up": "Look up",
    "down": "Look down",
    "left": "Look left",
    "right": "Look right",
}


@dataclasses.dataclass
class Job:
    id: str
    kind: str
    status: str
    created_at: str
    run_dir: str
    log_path: str
    command: list[str]
    outputs: dict[str, str]
    returncode: int | None = None
    error: str = ""
    process: subprocess.Popen[str] | None = dataclasses.field(default=None, repr=False)


jobs: dict[str, Job] = {}
jobs_lock = threading.Lock()
worker_lock = threading.RLock()
generation_worker: dict[str, Any] = {}


class AdoptedProcess:
    def __init__(self, pid: int) -> None:
        self.pid = int(pid)
        self.returncode: int | None = None

    def poll(self) -> int | None:
        try:
            os.kill(self.pid, 0)
        except ProcessLookupError:
            self.returncode = 0 if self.returncode is None else self.returncode
            return self.returncode
        except PermissionError:
            return None
        return None

    def terminate(self) -> None:
        os.kill(self.pid, signal.SIGTERM)


def _now_id() -> str:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{secrets.token_hex(3)}"


def _json_response(handler: BaseHTTPRequestHandler, data: Any, status: int = 200) -> None:
    payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


def _text_response(handler: BaseHTTPRequestHandler, text: str, content_type: str = "text/html; charset=utf-8") -> None:
    payload = text.encode("utf-8")
    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


def _error_response(handler: BaseHTTPRequestHandler, message: str, status: int = 400) -> None:
    _json_response(handler, {"ok": False, "error": message}, status=status)


def _safe_relpath(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    resolved = candidate.resolve()
    root = ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"path is outside the Lyra-2 workspace: {path}")
    return resolved


def _file_url(path: str | Path) -> str:
    resolved = _safe_relpath(path)
    rel = resolved.relative_to(ROOT.resolve()).as_posix()
    return "/file?path=" + urllib.parse.quote(rel)


def _read_tail(path: str | Path, limit: int = 24000) -> str:
    log_path = Path(path)
    if not log_path.is_file():
        return ""
    with log_path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - limit))
        return f.read().decode("utf-8", errors="replace")


def _append_log(path: str | Path, line: str) -> None:
    with Path(path).open("a", encoding="utf-8", errors="replace") as f:
        f.write(line)
        if not line.endswith("\n"):
            f.write("\n")


def _write_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(output_path)


def _load_image_size(path: Path) -> tuple[int, int]:
    if Image is not None:
        with Image.open(path) as img:
            return img.size

    suffix = path.suffix.lower()
    data = path.read_bytes()
    if suffix == ".png" and data[:8] == b"\x89PNG\r\n\x1a\n":
        width = int.from_bytes(data[16:20], "big")
        height = int.from_bytes(data[20:24], "big")
        return width, height
    raise RuntimeError("Pillow is not available and image size could not be inferred.")


def _camera_w2c(camera_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target.astype(np.float64) - camera_pos.astype(np.float64)
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-8:
        forward = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        forward = forward / forward_norm

    up_seed = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(forward, up_seed))) > 0.98:
        up_seed = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    right = np.cross(up_seed, forward)
    right /= max(np.linalg.norm(right), 1e-8)
    up = np.cross(forward, right)
    up /= max(np.linalg.norm(up), 1e-8)

    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = np.stack([right, up, forward], axis=0).astype(np.float32)
    w2c[:3, 3] = (-w2c[:3, :3] @ camera_pos.astype(np.float32)).astype(np.float32)
    return w2c


def _direction_vectors(yaw: float, pitch: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cp = math.cos(pitch)
    forward = np.array([math.sin(yaw) * cp, math.sin(pitch), math.cos(yaw) * cp], dtype=np.float64)
    forward /= max(np.linalg.norm(forward), 1e-8)
    up_seed = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(up_seed, forward)
    right /= max(np.linalg.norm(right), 1e-8)
    up = np.cross(forward, right)
    up /= max(np.linalg.norm(up), 1e-8)
    return forward, right, up


def _frames_for_seconds(seconds: float, fps: int) -> int:
    target = max(float(seconds) * int(fps), CHUNK_SIZE)
    chunks = max(1, int(round(target / CHUNK_SIZE)))
    return 1 + chunks * CHUNK_SIZE


def _intrinsics(width: int, height: int, num_frames: int) -> np.ndarray:
    focal = float(max(width, height)) * 1.5
    K = np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return np.repeat(K[None, :, :], num_frames, axis=0)


def _preset_pose_sequence(name: str, num_frames: int, strength: float) -> list[np.ndarray]:
    t = np.linspace(0.0, 1.0, num_frames, dtype=np.float64)
    strength = max(0.05, float(strength))
    matrices: list[np.ndarray] = []

    if name in {"orbit_left", "orbit_right"}:
        target = np.array([0.0, 0.0, 1.35], dtype=np.float64)
        radius = 1.35
        sign = 1.0 if name == "orbit_left" else -1.0
        angle = math.radians(42.0) * strength
        for value in t:
            theta = math.pi + sign * angle * float(value)
            center = np.array([radius * math.sin(theta), 0.0, radius + radius * math.cos(theta)], dtype=np.float64)
            matrices.append(_camera_w2c(center, target))
        return matrices

    if name == "spiral_forward":
        for value in t:
            theta = 2.0 * math.pi * 2.0 * float(value)
            center = np.array(
                [
                    0.16 * math.cos(theta) * strength,
                    0.08 * math.sin(theta) * strength,
                    0.75 * float(value) * strength,
                ],
                dtype=np.float64,
            )
            target = np.array([0.0, 0.0, 1.25 + 0.45 * float(value) * strength], dtype=np.float64)
            matrices.append(_camera_w2c(center, target))
        return matrices

    if name == "rotate_zoom":
        for value in t:
            yaw = math.radians(24.0) * strength * float(value)
            pitch = math.radians(3.0) * math.sin(math.pi * float(value))
            center = np.array([0.0, 0.0, 0.68 * float(value) * strength], dtype=np.float64)
            forward, _, _ = _direction_vectors(yaw, pitch)
            matrices.append(_camera_w2c(center, center + forward))
        return matrices

    endpoints = {
        "forward": (np.array([0.0, 0.0, 0.85]), 0.0, 0.0),
        "backward": (np.array([0.0, 0.0, -0.55]), 0.0, 0.0),
        "slide_left": (np.array([-0.65, 0.0, 0.18]), math.radians(-4.0), 0.0),
        "slide_right": (np.array([0.65, 0.0, 0.18]), math.radians(4.0), 0.0),
        "pan_left": (np.array([0.0, 0.0, 0.0]), math.radians(-32.0), 0.0),
        "pan_right": (np.array([0.0, 0.0, 0.0]), math.radians(32.0), 0.0),
    }
    center_end, yaw_end, pitch_end = endpoints.get(name, endpoints["forward"])
    center_end = center_end.astype(np.float64) * strength
    yaw_end *= strength
    pitch_end *= strength
    for value in t:
        center = center_end * float(value)
        yaw = yaw_end * float(value)
        pitch = pitch_end * float(value)
        forward, _, _ = _direction_vectors(yaw, pitch)
        matrices.append(_camera_w2c(center, center + forward))
    return matrices


def _parse_ops(raw: str) -> list[str]:
    text = (raw or "").replace(",", " ").replace("\n", " ")
    aliases = {
        "arrowup": "up",
        "arrowdown": "down",
        "arrowleft": "left",
        "arrowright": "right",
        "↑": "up",
        "↓": "down",
        "←": "left",
        "→": "right",
    }
    ops = []
    for token in text.split():
        token = token.strip().lower()
        token = aliases.get(token, token)
        if token in OP_LABELS:
            ops.append(token)
    return ops


def _keyboard_pose_sequence(raw_ops: str, num_frames: int, strength: float) -> tuple[list[np.ndarray], list[str]]:
    ops = _parse_ops(raw_ops)
    if not ops:
        ops = ["w", "w", "left", "w"]

    move_step = 0.28 * max(0.05, float(strength))
    rot_step = math.radians(12.0) * max(0.05, float(strength))
    centers = [np.array([0.0, 0.0, 0.0], dtype=np.float64)]
    yaws = [0.0]
    pitches = [0.0]

    center = centers[0].copy()
    yaw = 0.0
    pitch = 0.0
    for op in ops:
        forward, right, _up = _direction_vectors(yaw, pitch)
        if op == "w":
            center = center + forward * move_step
        elif op == "s":
            center = center - forward * move_step
        elif op == "a":
            center = center - right * move_step
        elif op == "d":
            center = center + right * move_step
        elif op == "left":
            yaw -= rot_step
        elif op == "right":
            yaw += rot_step
        elif op == "up":
            pitch = max(math.radians(-45.0), pitch - rot_step)
        elif op == "down":
            pitch = min(math.radians(45.0), pitch + rot_step)
        centers.append(center.copy())
        yaws.append(yaw)
        pitches.append(pitch)

    key_t = np.linspace(0.0, 1.0, len(centers), dtype=np.float64)
    frame_t = np.linspace(0.0, 1.0, num_frames, dtype=np.float64)
    centers_np = np.stack(centers, axis=0)
    interp_centers = np.stack([np.interp(frame_t, key_t, centers_np[:, axis]) for axis in range(3)], axis=1)
    interp_yaws = np.interp(frame_t, key_t, np.array(yaws, dtype=np.float64))
    interp_pitches = np.interp(frame_t, key_t, np.array(pitches, dtype=np.float64))

    matrices = []
    for center_i, yaw_i, pitch_i in zip(interp_centers, interp_yaws, interp_pitches):
        forward, _, _ = _direction_vectors(float(yaw_i), float(pitch_i))
        matrices.append(_camera_w2c(center_i, center_i + forward))
    return matrices, ops


def _compose_trajectory_start(matrices: list[np.ndarray], start_w2c: np.ndarray | None) -> list[np.ndarray]:
    if start_w2c is None or not matrices:
        return matrices
    base_start_c2w = np.linalg.inv(matrices[0].astype(np.float64))
    base_start_c2w_inv = np.linalg.inv(base_start_c2w)
    target_start_c2w = np.linalg.inv(start_w2c.astype(np.float64))
    composed: list[np.ndarray] = []
    for w2c_i in matrices:
        rel_c2w = base_start_c2w_inv @ np.linalg.inv(w2c_i.astype(np.float64))
        composed.append(np.linalg.inv(target_start_c2w @ rel_c2w).astype(np.float32))
    return composed


def _write_trajectory(
    output_path: Path,
    *,
    width: int,
    height: int,
    num_frames: int,
    mode: str,
    preset: str,
    operations: str,
    strength: float,
    start_w2c: np.ndarray | None = None,
) -> list[str]:
    if mode == "keyboard":
        matrices, ops = _keyboard_pose_sequence(operations, num_frames, strength)
    else:
        preset_name = preset if preset in PRESET_TRAJECTORIES else "forward"
        matrices = _preset_pose_sequence(preset_name, num_frames, strength)
        ops = [preset_name]

    matrices = _compose_trajectory_start(matrices, start_w2c)
    w2c = np.stack(matrices, axis=0).astype(np.float32)
    intrinsics = _intrinsics(width, height, num_frames)
    np.savez(
        output_path,
        w2c=w2c,
        intrinsics=intrinsics,
        image_height=np.array(height, dtype=np.int32),
        image_width=np.array(width, dtype=np.int32),
    )
    return ops


def _write_captions(path: Path, prompt: str, num_frames: int) -> None:
    prompt = (prompt or "").strip() or DEFAULT_PROMPT
    captions = {str(frame): prompt for frame in range(0, num_frames, CHUNK_SIZE)}
    if "0" not in captions:
        captions["0"] = prompt
    path.write_text(json.dumps(captions, ensure_ascii=False, indent=2), encoding="utf-8")


def _sample_images() -> list[dict[str, str]]:
    assets_dir = ROOT / "assets"
    if not assets_dir.is_dir():
        return []

    items = []
    image_exts = {".png", ".jpg", ".jpeg"}
    for path in sorted(p for p in assets_dir.rglob("*") if p.suffix.lower() in image_exts):
        txt_path = path.with_suffix(".txt")
        prompt = ""
        if txt_path.is_file():
            prompt = txt_path.read_text(encoding="utf-8", errors="replace").strip()
        else:
            captions_path = path.parent / "captions.json"
            if captions_path.is_file():
                try:
                    captions = json.loads(captions_path.read_text(encoding="utf-8", errors="replace"))
                    if isinstance(captions, dict) and captions:
                        prompt = str(captions.get("0") or captions[sorted(captions, key=lambda key: int(key))[0]]).strip()
                except Exception:
                    prompt = ""

        rel_path = path.relative_to(ROOT).as_posix()
        display_name = path.relative_to(assets_dir).as_posix()
        items.append(
            {
                "name": display_name,
                "path": rel_path,
                "url": _file_url(path),
                "prompt": prompt,
            }
        )
    return items


def _load_ui_settings() -> dict[str, Any]:
    if not SETTINGS_FILE.is_file():
        return {}
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_ui_settings(settings: dict[str, Any]) -> None:
    try:
        _write_json(SETTINGS_FILE, settings)
    except Exception:
        pass


def _field_to_text(field: cgi.FieldStorage | None, default: str = "") -> str:
    if field is None:
        return default
    if isinstance(field, list):
        field = field[0]
    value = field.value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _field_to_bool(field: cgi.FieldStorage | None, default: bool = False) -> bool:
    text = _field_to_text(field, "1" if default else "0").strip().lower()
    return text in {"1", "true", "yes", "on"}


def _field_to_int(field: cgi.FieldStorage | None, default: int) -> int:
    try:
        return int(float(_field_to_text(field, str(default))))
    except Exception:
        return default


def _field_to_float(field: cgi.FieldStorage | None, default: float) -> float:
    try:
        return float(_field_to_text(field, str(default)))
    except Exception:
        return default


def _sanitize_filename(name: str, default: str = "first_frame.png") -> str:
    clean = "".join(ch for ch in name if ch.isalnum() or ch in {"-", "_", "."}).strip(".")
    if not clean:
        clean = default
    suffix = Path(clean).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg"}:
        clean += ".png"
    return clean


def _copy_or_save_image(form: cgi.FieldStorage, input_dir: Path) -> tuple[Path, str]:
    image_field = form["image"] if "image" in form else None
    if image_field is not None and not isinstance(image_field, list) and image_field.filename:
        filename = _sanitize_filename(image_field.filename)
        suffix = Path(filename).suffix.lower()
        image_path = input_dir / f"first_frame{suffix}"
        with image_path.open("wb") as f:
            shutil.copyfileobj(image_field.file, f)
        return image_path, ""

    sample_rel = _field_to_text(form["sample_image"] if "sample_image" in form else None, "").strip()
    if sample_rel:
        sample_path = _safe_relpath(sample_rel)
        if not sample_path.is_file():
            raise FileNotFoundError(f"sample image not found: {sample_rel}")
        suffix = sample_path.suffix.lower()
        image_path = input_dir / f"first_frame{suffix}"
        shutil.copy2(sample_path, image_path)
        prompt = ""
        txt_path = sample_path.with_suffix(".txt")
        if txt_path.is_file():
            prompt = txt_path.read_text(encoding="utf-8", errors="replace").strip()
        return image_path, prompt

    raise ValueError("please upload an image or select a sample image")


def _read_job_metadata(job: Job) -> dict[str, Any]:
    path = Path(job.run_dir) / "metadata.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _path_if_file(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_file() else None


def _find_trajectory_for_job(job: Job) -> Path | None:
    path = _path_if_file(job.outputs.get("trajectory"))
    if path is not None:
        return path
    metadata = _read_job_metadata(job)
    return _path_if_file(str(metadata.get("trajectory", "")))


def _find_segment_video_for_job(job: Job) -> Path | None:
    path = _path_if_file(job.outputs.get("generated_video")) or _path_if_file(job.outputs.get("expected_video"))
    if path is not None:
        return path
    run_dir = Path(job.run_dir)
    candidates = [
        p for p in sorted(run_dir.rglob("*.mp4"))
        if p.name != AR_SEQUENCE_VIDEO_NAME and "_gs_ours" not in p.as_posix()
    ]
    return candidates[-1] if candidates else None


def _find_autoregressive_video_for_job(job: Job) -> Path | None:
    _discover_outputs(job)
    return (
        _path_if_file(job.outputs.get("autoregressive_video"))
        or _path_if_file(job.outputs.get("expected_autoregressive_video"))
        or _find_segment_video_for_job(job)
    )


def _job_pose_scale(job: Job, default: float) -> float:
    metadata = _read_job_metadata(job)
    try:
        return float(metadata.get("pose_scale", default))
    except Exception:
        return default


def _last_w2c_for_continuation(traj_path: Path, previous_pose_scale: float, current_pose_scale: float) -> np.ndarray:
    data = np.load(traj_path)
    w2c = data["w2c"][-1].astype(np.float32).copy()
    if abs(float(current_pose_scale)) > 1e-8:
        w2c[:3, 3] *= float(previous_pose_scale) / float(current_pose_scale)
    return w2c


def _require_cv2() -> Any:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for autoregressive continuation video handling.")
    return cv2


def _extract_last_video_frame(video_path: Path, output_path: Path) -> None:
    cv = _require_cv2()
    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"failed to open video: {video_path}")
    try:
        frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT) or 0)
        frame = None
        if frame_count > 0:
            cap.set(cv.CAP_PROP_POS_FRAMES, max(0, frame_count - 1))
            ok, candidate = cap.read()
            if ok and candidate is not None:
                frame = candidate
        if frame is None:
            cap.set(cv.CAP_PROP_POS_FRAMES, 0)
            while True:
                ok, candidate = cap.read()
                if not ok or candidate is None:
                    break
                frame = candidate
        if frame is None:
            raise RuntimeError(f"no frames found in video: {video_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv.imwrite(str(output_path), frame):
            raise RuntimeError(f"failed to save continuation frame: {output_path}")
    finally:
        cap.release()


def _concat_videos_cv2(video_paths: list[Path], output_path: Path, fps: float) -> None:
    cv = _require_cv2()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp.mp4")
    if tmp_path.exists():
        tmp_path.unlink()

    import imageio.v2 as imageio

    writer = imageio.get_writer(
        str(tmp_path),
        fps=float(max(fps, 1.0)),
        codec="libx264",
        macro_block_size=1,
        ffmpeg_params=["-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    frame_size: tuple[int, int] | None = None
    frames_written = 0
    try:
        for video_idx, video_path in enumerate(video_paths):
            cap = cv.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise FileNotFoundError(f"failed to open video: {video_path}")
            try:
                if video_idx > 0:
                    cap.set(cv.CAP_PROP_POS_FRAMES, 1)
                while True:
                    ok, frame_bgr = cap.read()
                    if not ok or frame_bgr is None:
                        break
                    h, w = frame_bgr.shape[:2]
                    if frame_size is None:
                        frame_size = (w, h)
                    elif (w, h) != frame_size:
                        frame_bgr = cv.resize(frame_bgr, frame_size, interpolation=cv.INTER_LINEAR)
                    frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
                    writer.append_data(frame_rgb)
                    frames_written += 1
            finally:
                cap.release()
    finally:
        writer.close()
    if frames_written <= 0:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise RuntimeError("no frames written while concatenating autoregressive video")
    tmp_path.replace(output_path)


def _ensure_autoregressive_video(job: Job) -> None:
    metadata = _read_job_metadata(job)
    if not metadata.get("continuation"):
        return
    existing = _path_if_file(job.outputs.get("autoregressive_video"))
    if existing is not None:
        return
    target = Path(
        str(
            metadata.get("autoregressive_video")
            or job.outputs.get("expected_autoregressive_video")
            or (Path(job.run_dir) / "videos" / AR_SEQUENCE_VIDEO_NAME)
        )
    )
    if target.is_file():
        job.outputs["autoregressive_video"] = str(target)
        return
    base = _path_if_file(str(metadata.get("continuation_base_video", "")))
    segment = _find_segment_video_for_job(job)
    if base is None or segment is None:
        return
    try:
        _concat_videos_cv2([base, segment], target, fps=float(metadata.get("fps", 16)))
        job.outputs["autoregressive_video"] = str(target)
        _append_log(job.log_path, f"[ui] autoregressive sequence video: {target}")
    except Exception as exc:
        _append_log(job.log_path, f"[ui] failed to build autoregressive sequence video: {exc}")


def _discover_outputs(job: Job) -> None:
    run_dir = Path(job.run_dir)
    if not run_dir.exists():
        return
    mp4s = sorted(run_dir.rglob("*.mp4"), key=lambda p: p.stat().st_mtime if p.exists() else 0)
    generated_candidates: list[Path] = []
    for path in mp4s:
        name = path.name
        if name == "gs_trajectory.mp4":
            job.outputs["reconstruction_video"] = str(path)
        elif name == AR_SEQUENCE_VIDEO_NAME:
            job.outputs["autoregressive_video"] = str(path)
        elif "_gs_ours" not in path.as_posix():
            generated_candidates.append(path)
    expected_video = _path_if_file(job.outputs.get("expected_video"))
    if expected_video is not None:
        job.outputs["generated_video"] = str(expected_video)
    elif generated_candidates:
        job.outputs["generated_video"] = str(generated_candidates[-1])
    step1_plys = sorted(run_dir.rglob("*_sparse_cache.ply"), key=lambda p: p.stat().st_mtime if p.exists() else 0)
    if step1_plys:
        job.outputs["step1_sparse_cache_ply"] = str(step1_plys[-1])
    plys = sorted(run_dir.rglob("reconstructed_scene.ply"), key=lambda p: p.stat().st_mtime if p.exists() else 0)
    if plys:
        job.outputs["ply"] = str(plys[-1])
    _ensure_autoregressive_video(job)


def _run_subprocess(job: Job) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    with jobs_lock:
        job.status = "running"

    _append_log(job.log_path, f"$ {' '.join(job.command)}")
    _append_log(job.log_path, "")
    try:
        process = subprocess.Popen(
            job.command,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        with jobs_lock:
            job.process = process
        assert process.stdout is not None
        for line in process.stdout:
            _append_log(job.log_path, line)
        returncode = process.wait()
        with jobs_lock:
            job.returncode = returncode
            job.process = None
            _discover_outputs(job)
            job.status = "complete" if returncode == 0 else "failed"
        if returncode != 0:
            _append_log(job.log_path, f"\nProcess exited with code {returncode}")
    except Exception as exc:
        with jobs_lock:
            job.status = "failed"
            job.error = str(exc)
            job.process = None
        _append_log(job.log_path, traceback.format_exc())


def _register_job(kind: str, run_dir: Path, command: list[str]) -> Job:
    job = Job(
        id=run_dir.name,
        kind=kind,
        status="queued",
        created_at=dt.datetime.now().isoformat(timespec="seconds"),
        run_dir=str(run_dir),
        log_path=str(run_dir / f"{kind}.log"),
        command=command,
        outputs={},
    )
    with jobs_lock:
        jobs[job.id] = job
    threading.Thread(target=_run_subprocess, args=(job,), daemon=True).start()
    return job


def _worker_config_from_form(form: cgi.FieldStorage) -> dict[str, Any]:
    return {
        "experiment": _field_to_text(form["experiment"] if "experiment" in form else None, "lyra2"),
        "checkpoint_dir": _field_to_text(form["checkpoint_dir"] if "checkpoint_dir" in form else None, "checkpoints/model"),
        "use_dmd": _field_to_bool(form["use_dmd"] if "use_dmd" in form else None, True),
        "use_moge_scale": _field_to_bool(form["use_moge_scale"] if "use_moge_scale" in form else None, True),
        "da3_model_name": _field_to_text(form["da3_model_name"] if "da3_model_name" in form else None, "depth-anything/DA3NESTED-GIANT-LARGE-1.1"),
        "da3_model_path_custom": _field_to_text(form["da3_model_path_custom"] if "da3_model_path_custom" in form else None, "") or None,
        "export_sparse_cache_ply": _field_to_bool(form["export_sparse_cache_ply"] if "export_sparse_cache_ply" in form else None, True),
        "sparse_cache_max_points": _field_to_int(form["sparse_cache_max_points"] if "sparse_cache_max_points" in form else None, 200000),
    }


def _worker_signature(config: dict[str, Any]) -> dict[str, Any]:
    keys = ["experiment", "checkpoint_dir", "use_dmd", "use_moge_scale", "da3_model_name", "da3_model_path_custom"]
    return {key: config.get(key) for key in keys}


def _read_worker_status() -> dict[str, Any]:
    with worker_lock:
        state = dict(generation_worker)
    process = state.get("process")
    status = {
        "enabled": process is not None,
        "running": bool(process is not None and process.poll() is None),
        "status": "stopped",
        "config": state.get("config"),
        "control_dir": str(state.get("control_dir")) if state.get("control_dir") else "",
        "log": "",
    }
    control_dir = state.get("control_dir")
    if control_dir:
        status_path = Path(control_dir) / "status.json"
        if status_path.is_file():
            try:
                status.update(json.loads(status_path.read_text(encoding="utf-8")))
            except Exception as exc:
                status["status"] = "unknown"
                status["error"] = str(exc)
        worker_log = Path(control_dir) / "worker.log"
        status["log"] = _read_tail(worker_log, limit=12000)
    if process is not None and process.poll() is None and status.get("status") == "stopped":
        status["status"] = "starting"
        status["message"] = "starting preload worker"
    if process is not None and process.poll() is not None and status.get("status") not in {"failed", "stopped"}:
        status["status"] = "exited"
        status["returncode"] = process.returncode
    return status


def _pid_is_running(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
    except (TypeError, ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    return True


def _adopt_existing_worker() -> None:
    base = RUNS_DIR / "preload_workers"
    if not base.is_dir():
        return
    with worker_lock:
        old_process = generation_worker.get("process")
        if old_process is not None and old_process.poll() is None:
            return
        candidates = sorted(base.glob("*/status.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for status_path in candidates:
            try:
                payload = json.loads(status_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            pid = payload.get("pid")
            if not _pid_is_running(pid):
                continue
            state = str(payload.get("status", ""))
            if state not in {"starting", "loading", "ready", "busy", "offloaded", "offloading", "moving_gpu"}:
                continue
            control_dir = status_path.parent
            generation_worker.clear()
            generation_worker.update({
                "process": AdoptedProcess(int(pid)),
                "config": payload.get("config") or {},
                "control_dir": control_dir,
                "command": ["adopted-preloaded-worker", str(control_dir)],
            })
            return


def _start_preload_worker(form: cgi.FieldStorage) -> dict[str, Any]:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    config = _worker_config_from_form(form)
    signature = _worker_signature(config)

    with worker_lock:
        old_process = generation_worker.get("process")
        old_config = generation_worker.get("config")
        old_signature = _worker_signature(old_config or {}) if old_config else None
        if old_process is not None and old_process.poll() is None and old_signature == signature:
            return _read_worker_status()
        if old_process is not None and old_process.poll() is None:
            try:
                os.killpg(old_process.pid, signal.SIGTERM)
            except Exception:
                old_process.terminate()

        control_dir = RUNS_DIR / "preload_workers" / _now_id()
        control_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["PYTHONPATH"] = "."
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        log_file = (control_dir / "worker.log").open("a", encoding="utf-8", errors="replace")
        command = [
            sys.executable,
            "lyra2_generation_worker.py",
            "--control-dir",
            str(control_dir),
            "--config-json",
            json.dumps(config, ensure_ascii=False),
        ]
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        log_file.close()
        generation_worker.clear()
        generation_worker.update({
            "process": process,
            "config": config,
            "control_dir": control_dir,
            "command": command,
        })
    return _read_worker_status()


def _worker_device_command(action: str, timeout: float = 900.0) -> dict[str, Any]:
    status = _read_worker_status()
    if not status.get("running"):
        raise RuntimeError("preloaded worker is not running")
    if status.get("status") == "busy":
        raise RuntimeError("preloaded worker is busy")
    control_dir_text = str(status.get("control_dir") or "")
    if not control_dir_text:
        raise RuntimeError("preloaded worker control directory is missing")
    control_dir = Path(control_dir_text)
    command_id = _now_id()
    command_path = control_dir / "commands" / f"{command_id}.json"
    result_path = control_dir / "command_results" / f"{command_id}.json"
    command = {"command_id": command_id, "action": action, "created_at": time.time()}
    _write_json(command_path, command)
    started = time.time()
    while time.time() - started < timeout:
        if result_path.is_file():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            finally:
                try:
                    result_path.unlink()
                except FileNotFoundError:
                    pass
            if not result.get("ok"):
                raise RuntimeError(result.get("error") or f"worker command failed: {action}")
            result["wait_elapsed_sec"] = time.time() - started
            refreshed = _read_worker_status()
            refreshed["command_result"] = result
            return refreshed
        refreshed = _read_worker_status()
        if not refreshed.get("running"):
            raise RuntimeError("preloaded worker exited before command finished")
        time.sleep(0.25)
    raise TimeoutError(f"worker command timed out: {action}")


def _offload_worker_to_cpu() -> dict[str, Any]:
    return _worker_device_command("offload_cpu")


def _move_worker_to_gpu() -> dict[str, Any]:
    return _worker_device_command("move_gpu")


def _release_preload_worker() -> dict[str, Any]:
    with worker_lock:
        process = generation_worker.get("process")
        control_dir = generation_worker.get("control_dir")
        config = generation_worker.get("config") or {}
        pid = getattr(process, "pid", None)
        running = bool(process is not None and process.poll() is None)
        worker_log = Path(control_dir) / "worker.log" if control_dir else None
        if worker_log is not None:
            _append_log(worker_log, "[ui] release requested; stopping preloaded worker")
        if control_dir:
            _write_json(Path(control_dir) / "status.json", {
                "status": "released",
                "updated_at": time.time(),
                "pid": pid,
                "config": config,
                "message": "models released by UI",
            })
        if running and pid is not None:
            try:
                os.killpg(int(pid), signal.SIGTERM)
            except Exception:
                process.terminate()
            if hasattr(process, "wait"):
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(int(pid), signal.SIGKILL)
                    except Exception:
                        process.kill()
                    process.wait(timeout=2)
            else:
                for _ in range(20):
                    if not _pid_is_running(pid):
                        break
                    time.sleep(0.1)
                if _pid_is_running(pid):
                    try:
                        os.killpg(int(pid), signal.SIGKILL)
                    except Exception:
                        pass
        generation_worker.clear()
    return {
        "enabled": False,
        "running": False,
        "status": "stopped",
        "message": "models released",
        "config": config,
        "control_dir": str(control_dir) if control_dir else "",
        "released_pid": pid,
        "log": _read_tail(worker_log, limit=12000) if worker_log is not None else "",
    }


def _worker_ready_for(config: dict[str, Any]) -> bool:
    status = _read_worker_status()
    if status.get("status") != "ready" or not status.get("running"):
        return False
    worker_config = status.get("config") or {}
    return _worker_signature(worker_config) == _worker_signature(config)


def _watch_worker_job(job: Job, result_path: Path) -> None:
    with jobs_lock:
        job.status = "queued"
    while True:
        if result_path.is_file():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except Exception as exc:
                result = {"ok": False, "error": str(exc)}
            with jobs_lock:
                if result.get("ok"):
                    job.outputs.update({k: str(v) for k, v in (result.get("outputs") or {}).items()})
                    _discover_outputs(job)
                    job.status = "complete"
                    job.returncode = 0
                else:
                    job.status = "failed"
                    job.returncode = 1
                    job.error = result.get("error", "worker task failed")
            return
        status = _read_worker_status()
        with jobs_lock:
            if status.get("current_job_id") == job.id and job.status != "cancelled":
                job.status = "running"
            if job.status == "cancelled":
                return
        if not status.get("running"):
            with jobs_lock:
                job.status = "failed"
                job.returncode = 1
                job.error = "preloaded worker exited before the task finished"
            _append_log(job.log_path, job.error)
            return
        time.sleep(1.0)


def _register_worker_generation_job(run_dir: Path, command: list[str], task: dict[str, Any], outputs: dict[str, str]) -> Job:
    status = _read_worker_status()
    control_dir = Path(status["control_dir"])
    job = Job(
        id=run_dir.name,
        kind="generation",
        status="queued",
        created_at=dt.datetime.now().isoformat(timespec="seconds"),
        run_dir=str(run_dir),
        log_path=str(run_dir / "generation.log"),
        command=command,
        outputs=dict(outputs),
    )
    task = dict(task)
    task["job_id"] = job.id
    task["log_path"] = job.log_path
    task_path = control_dir / "tasks" / f"{job.id}.json"
    result_path = control_dir / "results" / f"{job.id}.json"
    task_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    _append_log(job.log_path, f"[ui] queued on preloaded worker: {control_dir}")
    _write_json(task_path, task)
    with jobs_lock:
        jobs[job.id] = job
    threading.Thread(target=_watch_worker_job, args=(job, result_path), daemon=True).start()
    return job


def _start_generation(form: cgi.FieldStorage) -> Job:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = RUNS_DIR / _now_id()
    input_dir = run_dir / "input"
    video_dir = run_dir / "videos"
    input_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    fps = max(1, _field_to_int(form["fps"] if "fps" in form else None, 16))
    seconds = max(1.0, _field_to_float(form["duration"] if "duration" in form else None, 5.0))
    num_frames = _frames_for_seconds(seconds, fps)
    trajectory_mode = _field_to_text(form["trajectory_mode"] if "trajectory_mode" in form else None, "preset")
    preset = _field_to_text(form["preset"] if "preset" in form else None, "forward")
    operations = _field_to_text(form["operations"] if "operations" in form else None, "")
    strength = _field_to_float(form["strength"] if "strength" in form else None, 1.0)

    checkpoint_dir = _field_to_text(form["checkpoint_dir"] if "checkpoint_dir" in form else None, "checkpoints/model")
    experiment = _field_to_text(form["experiment"] if "experiment" in form else None, "lyra2")
    resolution = _field_to_text(form["resolution"] if "resolution" in form else None, "480,832")
    pose_scale = _field_to_float(form["pose_scale"] if "pose_scale" in form else None, 1.1)
    seed = _field_to_int(form["seed"] if "seed" in form else None, 1)
    steps = _field_to_int(form["sampling_steps"] if "sampling_steps" in form else None, 35)
    use_dmd = _field_to_bool(form["use_dmd"] if "use_dmd" in form else None, True)
    use_moge = _field_to_bool(form["use_moge_scale"] if "use_moge_scale" in form else None, True)
    export_sparse_cache = _field_to_bool(form["export_sparse_cache_ply"] if "export_sparse_cache_ply" in form else None, True)
    sparse_cache_max_points = _field_to_int(form["sparse_cache_max_points"] if "sparse_cache_max_points" in form else None, 200000)

    continue_enabled = _field_to_bool(form["ar_continue"] if "ar_continue" in form else None, True)
    continue_job_id = _field_to_text(form["continue_from_job"] if "continue_from_job" in form else None, "").strip()
    source_job: Job | None = None
    continuation_base_video: Path | None = None
    continuation_base_traj: Path | None = None
    continuation_start_w2c: np.ndarray | None = None
    sample_prompt = ""

    if continue_enabled and continue_job_id:
        with jobs_lock:
            source_job = jobs.get(continue_job_id)
        if source_job is None:
            raise ValueError(f"unknown continuation job id: {continue_job_id}")
        with jobs_lock:
            source_status = source_job.status
        if source_status != "complete":
            raise ValueError("previous generation is not complete yet")
        continuation_base_video = _find_autoregressive_video_for_job(source_job)
        continuation_base_traj = _find_trajectory_for_job(source_job)
        if continuation_base_video is None or not continuation_base_video.is_file():
            raise FileNotFoundError("previous generated video is not available for continuation")
        if continuation_base_traj is None or not continuation_base_traj.is_file():
            raise FileNotFoundError("previous trajectory is not available for continuation")
        image_path = input_dir / "first_frame.png"
        _extract_last_video_frame(continuation_base_video, image_path)
        previous_pose_scale = _job_pose_scale(source_job, pose_scale)
        continuation_start_w2c = _last_w2c_for_continuation(
            continuation_base_traj,
            previous_pose_scale=previous_pose_scale,
            current_pose_scale=pose_scale,
        )
        source_metadata = _read_job_metadata(source_job)
        sample_prompt = str(source_metadata.get("prompt", ""))
    else:
        image_path, sample_prompt = _copy_or_save_image(form, input_dir)

    width, height = _load_image_size(image_path)
    prompt = _field_to_text(form["prompt"] if "prompt" in form else None, "").strip() or sample_prompt or DEFAULT_PROMPT

    traj_path = input_dir / "trajectory.npz"
    captions_path = input_dir / "captions.json"
    used_ops = _write_trajectory(
        traj_path,
        width=width,
        height=height,
        num_frames=num_frames,
        mode="keyboard" if trajectory_mode == "keyboard" else "preset",
        preset=preset,
        operations=operations,
        strength=strength,
        start_w2c=continuation_start_w2c,
    )
    _write_captions(captions_path, prompt, num_frames)

    is_continuation = continuation_base_video is not None
    segment_video_path = video_dir / f"{image_path.stem}.mp4"
    ar_sequence_path = video_dir / AR_SEQUENCE_VIDEO_NAME if is_continuation else None
    source_metadata = _read_job_metadata(source_job) if source_job is not None else {}
    previous_segments_raw = source_metadata.get("scene_segment_videos") if isinstance(source_metadata, dict) else None
    previous_segments = [str(item) for item in previous_segments_raw] if isinstance(previous_segments_raw, list) else []
    if is_continuation and not previous_segments and source_job is not None:
        source_segment = _find_segment_video_for_job(source_job)
        if source_segment is not None:
            previous_segments = [str(source_segment)]
    scene_segment_videos = previous_segments + [str(segment_video_path)]
    current_pd_path = str(video_dir / f"{image_path.stem}_step1_sparse_cache.ply") if export_sparse_cache else ""
    previous_pd_raw = source_metadata.get("scene_segment_pd_clouds") if isinstance(source_metadata, dict) else None
    previous_pd_clouds = [str(item) for item in previous_pd_raw] if isinstance(previous_pd_raw, list) else []
    if is_continuation and not previous_pd_clouds and source_job is not None:
        source_outputs = dict(source_job.outputs)
        source_pd = source_outputs.get("step1_sparse_cache_ply") or source_outputs.get("expected_step1_sparse_cache_ply")
        if source_pd:
            previous_pd_clouds = [str(source_pd)]
    scene_segment_pd_clouds = previous_pd_clouds + [current_pd_path]
    scene_id = str(source_metadata.get("scene_id") or (source_job.id if source_job is not None else run_dir.name))
    scene_segment_index = len(scene_segment_videos)
    scene_full_video = ar_sequence_path or segment_video_path
    metadata = {
        "image": str(image_path),
        "prompt": prompt,
        "fps": fps,
        "seconds_requested": seconds,
        "num_frames": num_frames,
        "trajectory_mode": trajectory_mode,
        "preset": preset,
        "operations": used_ops,
        "strength": strength,
        "trajectory": str(traj_path),
        "captions": str(captions_path),
        "pose_scale": pose_scale,
        "continuation": is_continuation,
        "continue_from_job": continue_job_id if is_continuation else "",
        "continuation_base_video": str(continuation_base_video) if continuation_base_video else "",
        "continuation_base_trajectory": str(continuation_base_traj) if continuation_base_traj else "",
        "autoregressive_video": str(ar_sequence_path) if ar_sequence_path else "",
        "scene_id": scene_id,
        "scene_segment_index": scene_segment_index,
        "scene_segment_videos": scene_segment_videos,
        "scene_segment_pd_clouds": scene_segment_pd_clouds,
        "scene_full_video": str(scene_full_video),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    command = [
        sys.executable,
        "-m",
        "lyra_2._src.inference.lyra2_custom_traj_inference",
        "--input_image_path",
        str(image_path),
        "--trajectory_path",
        str(traj_path),
        "--captions_path",
        str(captions_path),
        "--prompt",
        prompt,
        "--experiment",
        experiment,
        "--checkpoint_dir",
        checkpoint_dir,
        "--output_path",
        str(video_dir),
        "--num_frames",
        str(num_frames),
        "--fps",
        str(fps),
        "--resolution",
        resolution,
        "--pose_scale",
        str(pose_scale),
        "--seed",
        str(seed),
        "--num_sampling_step",
        str(steps),
        "--num_samples",
        "1",
    ]
    if use_dmd:
        command.append("--use_dmd")
    if export_sparse_cache:
        command += ["--export_sparse_cache_ply", "--sparse_cache_max_points", str(sparse_cache_max_points)]
    if not use_moge:
        command.append("--no-use_moge_scale")

    expected_outputs = {
        "input_image": str(image_path),
        "trajectory": str(traj_path),
        "captions": str(captions_path),
        "expected_video": str(segment_video_path),
    }
    if is_continuation and ar_sequence_path is not None:
        expected_outputs["continuation_base_video"] = str(continuation_base_video)
        expected_outputs["expected_autoregressive_video"] = str(ar_sequence_path)
    if export_sparse_cache:
        expected_outputs["expected_step1_sparse_cache_ply"] = current_pd_path

    worker_config = _worker_config_from_form(form)
    if _worker_ready_for(worker_config):
        task = {
            "input_image_path": str(image_path),
            "trajectory_path": str(traj_path),
            "captions_path": str(captions_path),
            "prompt": prompt,
            "output_path": str(video_dir),
            "num_frames": num_frames,
            "fps": fps,
            "resolution": resolution,
            "pose_scale": pose_scale,
            "seed": seed,
            "num_sampling_step": steps,
            "export_sparse_cache_ply": export_sparse_cache,
            "sparse_cache_max_points": sparse_cache_max_points,
        }
        worker_status = _read_worker_status()
        worker_command = ["preloaded-worker", str(worker_status.get("control_dir", ""))]
        return _register_worker_generation_job(run_dir, worker_command, task, expected_outputs)

    job = _register_job("generation", run_dir, command)
    job.outputs.update(expected_outputs)
    return job


def _find_generated_video_for_job(job: Job) -> Path | None:
    path = _find_autoregressive_video_for_job(job)
    if path is not None and path.is_file():
        return path
    return _find_segment_video_for_job(job)


def _start_reconstruction(form: cgi.FieldStorage) -> Job:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    job_id = _field_to_text(form["job_id"] if "job_id" in form else None, "").strip()
    video_text = _field_to_text(form["video_path"] if "video_path" in form else None, "").strip()

    video_path: Path | None = None
    if video_text:
        video_path = _safe_relpath(video_text)
    elif job_id:
        with jobs_lock:
            source_job = jobs.get(job_id)
        if source_job is None:
            raise ValueError(f"unknown job id: {job_id}")
        video_path = _find_generated_video_for_job(source_job)
    if video_path is None or not video_path.is_file():
        raise FileNotFoundError("generated video is not available yet")

    run_dir = video_path.parent / f"{video_path.stem}_recon_job"
    run_dir.mkdir(parents=True, exist_ok=True)
    force = _field_to_bool(form["force"] if "force" in form else None, False)
    max_frames = _field_to_int(form["max_frames"] if "max_frames" in form else None, 0)
    da3_max_frames = _field_to_int(form["da3_max_frames"] if "da3_max_frames" in form else None, 128)

    command = [
        sys.executable,
        "-m",
        "lyra_2._src.inference.vipe_da3_gs_recon",
        "--input_video_path",
        str(video_path),
        "--da3_max_frames",
        str(da3_max_frames),
    ]
    if max_frames > 0:
        command += ["--max_frames", str(max_frames)]
    if force:
        command.append("--force")

    job = _register_job("reconstruction", run_dir, command)
    job.outputs["source_video"] = str(video_path)
    job.outputs["expected_reconstruction_video"] = str(video_path.parent / f"{video_path.stem}_gs_ours" / "gs_trajectory.mp4")
    job.outputs["expected_ply"] = str(video_path.parent / f"{video_path.stem}_gs_ours" / "reconstructed_scene.ply")
    return job


def _scene_payload(job: Job, outputs: dict[str, str]) -> dict[str, Any]:
    metadata = _read_job_metadata(job)
    scene_id = str(metadata.get("scene_id") or job.id)
    try:
        segment_index = int(metadata.get("scene_segment_index") or 1)
    except Exception:
        segment_index = 1
    raw_segments = metadata.get("scene_segment_videos")
    if isinstance(raw_segments, list):
        segment_paths = [str(item) for item in raw_segments]
    else:
        segment_paths = []
    if not segment_paths:
        base_segment = metadata.get("continuation_base_video")
        if base_segment:
            segment_paths.append(str(base_segment))
        fallback_segment = outputs.get("generated_video") or outputs.get("expected_video")
        if fallback_segment:
            segment_paths.append(str(fallback_segment))
    seen_segments: set[str] = set()
    segment_paths = [
        path for path in segment_paths
        if path and not (path in seen_segments or seen_segments.add(path))
    ]

    raw_pd_clouds = metadata.get("scene_segment_pd_clouds")
    if isinstance(raw_pd_clouds, list):
        pd_paths = [str(item) for item in raw_pd_clouds]
    else:
        pd_paths = []
    if not pd_paths:
        fallback_pd = outputs.get("step1_sparse_cache_ply") or outputs.get("expected_step1_sparse_cache_ply")
        if fallback_pd:
            pd_paths.append(str(fallback_pd))
    seen_pd: set[str] = set()
    pd_entries = [
        (idx, path) for idx, path in enumerate(pd_paths, start=1)
        if path and not (path in seen_pd or seen_pd.add(path))
    ]

    whole_path = (
        outputs.get("autoregressive_video")
        or outputs.get("expected_autoregressive_video")
        or metadata.get("scene_full_video")
        or outputs.get("generated_video")
        or outputs.get("expected_video")
    )
    segment_count = max(len(segment_paths), 1)
    segment_index = max(segment_index, segment_count)
    seconds = float(metadata.get("seconds_requested") or 5.0)

    def entry(label: str, kind: str, path_value: str, index: int | None = None) -> dict[str, Any]:
        item = {
            "label": label,
            "kind": kind,
            "path": str(path_value),
            "exists": False,
            "url": "",
        }
        if index is not None:
            item["index"] = index
        try:
            resolved = _safe_relpath(path_value)
            item["exists"] = resolved.is_file()
            if resolved.is_file():
                item["url"] = _file_url(resolved)
        except Exception:
            pass
        return item

    videos: list[dict[str, Any]] = []
    if whole_path:
        videos.append(entry(f"Full scene ({segment_count} segments)", "full", str(whole_path)))
    for idx, segment_path in enumerate(segment_paths, start=1):
        videos.append(entry(f"Segment {idx:02d} ({seconds:g}s)", "segment", segment_path, idx))

    pd_clouds = [entry(f"Segment {idx:02d} Point cloud", "pd", path, idx) for idx, path in pd_entries]

    gs_clouds: list[dict[str, Any]] = []
    gs_renders: list[dict[str, Any]] = []
    seen_gs: set[str] = set()
    seen_render: set[str] = set()

    def add_gs_for_video(label: str, video_path: str, index: int | None = None) -> None:
        if not video_path:
            return
        video = Path(str(video_path))
        gs_dir = video.parent / f"{video.stem}_gs_ours"
        ply_path = str(gs_dir / "reconstructed_scene.ply")
        render_path = str(gs_dir / "gs_trajectory.mp4")
        if ply_path not in seen_gs:
            seen_gs.add(ply_path)
            gs_clouds.append(entry(f"{label} GS", "gs", ply_path, index))
        if render_path not in seen_render:
            seen_render.add(render_path)
            gs_renders.append(entry(f"{label} GS_Render", "gs_render", render_path, index))

    for video_item in videos:
        add_gs_for_video(str(video_item.get("label") or "Scene"), str(video_item.get("path") or ""), video_item.get("index"))

    source_video = outputs.get("source_video")
    source_label = "Selected video"
    if source_video:
        for video_item in videos:
            if str(video_item.get("path") or "") == str(source_video):
                source_label = str(video_item.get("label") or source_label)
                break
        add_gs_for_video(source_label, str(source_video))

    output_ply = outputs.get("ply") or outputs.get("expected_ply")
    if output_ply and output_ply not in seen_gs:
        seen_gs.add(output_ply)
        gs_clouds.append(entry(f"{source_label} GS", "gs", str(output_ply)))
    output_render = outputs.get("reconstruction_video") or outputs.get("expected_reconstruction_video")
    if output_render and output_render not in seen_render:
        seen_render.add(output_render)
        gs_renders.append(entry(f"{source_label} GS_Render", "gs_render", str(output_render)))

    return {
        "id": scene_id,
        "segment_index": segment_index,
        "segment_count": segment_count,
        "is_continuation": bool(metadata.get("continuation")),
        "next_mode": "continue" if segment_count > 0 else "new",
        "videos": videos,
        "pd_clouds": pd_clouds,
        "gs_clouds": gs_clouds,
        "gs_renders": gs_renders,
    }


def _job_payload(job: Job) -> dict[str, Any]:
    with jobs_lock:
        _discover_outputs(job)
        outputs = dict(job.outputs)
        status = job.status
        returncode = job.returncode
        error = job.error
    urls = {}
    for key, value in outputs.items():
        try:
            path = _safe_relpath(value)
        except Exception:
            continue
        if path.exists():
            urls[key] = _file_url(path)
    return {
        "ok": True,
        "id": job.id,
        "kind": job.kind,
        "status": status,
        "created_at": job.created_at,
        "run_dir": job.run_dir,
        "command": job.command,
        "outputs": outputs,
        "urls": urls,
        "scene": _scene_payload(job, outputs) if job.kind == "generation" else {},
        "returncode": returncode,
        "error": error,
        "log": _read_tail(job.log_path),
    }


def _cancel_job(job_id: str) -> bool:
    with jobs_lock:
        job = jobs.get(job_id)
        process = job.process if job is not None else None
        if job is None:
            return False
        job.status = "cancelled"
    if process is not None and process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except Exception:
            process.terminate()
    _append_log(job.log_path, "Cancelled by user.")
    return True


PLY_DTYPES = {
    "char": "i1",
    "uchar": "u1",
    "int8": "i1",
    "uint8": "u1",
    "short": "i2",
    "ushort": "u2",
    "int16": "i2",
    "uint16": "u2",
    "int": "i4",
    "uint": "u4",
    "int32": "i4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}

GS_RASTER_CACHE: dict[tuple[str, int, int, int, int], bytes] = {}
GS_RASTER_CACHE_LOCK = threading.Lock()


def _gs_camera_view_hint(path: Path, center: np.ndarray, norm_scale: float) -> dict[str, Any] | None:
    cameras_path = path.parent / "cameras.npz"
    if not cameras_path.is_file() or norm_scale <= 1e-8:
        return None
    try:
        data = np.load(cameras_path)
        w2c = None
        for key in ("w2c_da3", "w2c_render", "w2c_vipe"):
            if key in data and data[key].ndim == 3 and data[key].shape[0] > 0:
                w2c = data[key][0].astype(np.float64)
                break
        if w2c is None:
            return None
        c2w = np.linalg.inv(w2c)
        eye_world = c2w[:3, 3]
        forward_world = c2w[:3, 2]
        up_world = c2w[:3, 1]

        flip = np.array([1.0, -1.0, 1.0], dtype=np.float64)
        preview_scale = 2.2 / float(norm_scale)

        def point_to_preview(value: np.ndarray) -> np.ndarray:
            return (value.astype(np.float64) - center.astype(np.float64)) * preview_scale * flip

        def dir_to_preview(value: np.ndarray) -> np.ndarray:
            out = value.astype(np.float64) * flip
            length = np.linalg.norm(out)
            if length <= 1e-8:
                return out
            return out / length

        forward = dir_to_preview(forward_world)
        up = dir_to_preview(up_world)
        if np.linalg.norm(forward) <= 1e-8 or np.linalg.norm(up) <= 1e-8:
            return None
        eye = point_to_preview(eye_world)
        # Pull the default view slightly behind the source camera so near splats are not clipped.
        eye = eye - forward * 0.035
        target = eye + forward

        fov_y = math.pi / 4.0
        for key in ("intrinsics_da3", "intrinsics_vipe"):
            if key in data and data[key].ndim == 3 and data[key].shape[0] > 0:
                K = data[key][0].astype(np.float64)
                fy = float(K[1, 1])
                cy = float(K[1, 2])
                if fy > 1e-6 and cy > 1e-6:
                    fov_y = 2.0 * math.atan(max(1.0, cy * 2.0) / (2.0 * fy))
                break
        fov_y = max(math.radians(22.0), min(math.radians(85.0), fov_y))
        return {
            "eye": eye.round(6).tolist(),
            "target": target.round(6).tolist(),
            "up": up.round(6).tolist(),
            "fov_y": round(float(fov_y), 6),
        }
    except Exception:
        return None



def _gs_raster_preview(path: Path, max_points: int = 700000, width: int = 832, height: int = 480) -> bytes:
    if Image is None:
        raise RuntimeError("Pillow is required for GS raster preview")
    if not path.is_file():
        raise FileNotFoundError(str(path))
    cameras_path = path.parent / "cameras.npz"
    if not cameras_path.is_file():
        raise FileNotFoundError(str(cameras_path))

    width = max(240, min(int(width), 1600))
    height = max(160, min(int(height), 1200))
    max_points = max(50000, min(int(max_points), 1000000))
    stat = path.stat()
    cache_key = (str(path), stat.st_mtime_ns, max_points, width, height)
    with GS_RASTER_CACHE_LOCK:
        cached = GS_RASTER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    with path.open("rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("invalid PLY: missing end_header")
            text = line.decode("ascii", errors="replace").strip()
            header_lines.append(text)
            if text == "end_header":
                break
        data_start = f.tell()

    fmt = "ascii"
    vertex_count = 0
    vertex_props: list[tuple[str, str]] = []
    in_vertex = False
    for line in header_lines:
        parts = line.split()
        if not parts:
            continue
        if parts[:2] == ["format", "binary_little_endian"]:
            fmt = "binary_little_endian"
        elif parts[:2] == ["format", "ascii"]:
            fmt = "ascii"
        elif parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                vertex_count = int(parts[2])
        elif parts[0] == "property" and in_vertex and len(parts) >= 3 and parts[1] != "list":
            vertex_props.append((parts[2], parts[1]))

    if fmt != "binary_little_endian":
        raise ValueError("GS raster preview currently expects binary_little_endian PLY")
    if vertex_count <= 0:
        raise ValueError("PLY has no vertices")

    dtype = np.dtype([(name, "<" + PLY_DTYPES.get(kind, "f4")) for name, kind in vertex_props])
    stride = max(1, math.ceil(vertex_count / max_points))
    vertices = np.memmap(path, dtype=dtype, mode="r", offset=data_start, shape=(vertex_count,))
    arr = np.asarray(vertices[::stride])
    names = arr.dtype.names or ()
    if not all(axis in names for axis in ("x", "y", "z")):
        raise ValueError("PLY is missing x/y/z properties")

    points = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)
    finite = np.all(np.isfinite(points), axis=1)
    if not np.any(finite):
        raise ValueError("no finite GS points")
    points = points[finite]

    if all(name in names for name in ("red", "green", "blue")):
        colors = np.stack([arr["red"], arr["green"], arr["blue"]], axis=1).astype(np.float32)[finite]
    elif all(name in names for name in ("f_dc_0", "f_dc_1", "f_dc_2")):
        sh0 = 0.28209479177387814
        fdc = np.stack([arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"]], axis=1).astype(np.float32)[finite]
        colors = np.clip((0.5 + sh0 * fdc) * 255.0 + 0.5, 0.0, 255.0)
    else:
        colors = np.full((points.shape[0], 3), 150.0, dtype=np.float32)

    if "opacity" in names:
        opacity_raw = arr["opacity"].astype(np.float32)[finite]
        opacities = 1.0 / (1.0 + np.exp(-np.clip(opacity_raw, -20.0, 20.0)))
    else:
        opacities = np.ones(points.shape[0], dtype=np.float32)

    scale_world = None
    if all(f"scale_{idx}" in names for idx in range(3)):
        scale_raw = np.stack([arr[f"scale_{idx}"] for idx in range(3)], axis=1).astype(np.float32)[finite]
        scale_world = np.exp(np.clip(scale_raw, -12.0, 4.0))
        world_radii = scale_world.mean(axis=1)
    else:
        world_radii = np.ones(points.shape[0], dtype=np.float32) * 0.01

    rotations = None
    if all(f"rot_{idx}" in names for idx in range(4)):
        rotations = np.stack([arr[f"rot_{idx}"] for idx in range(4)], axis=1).astype(np.float32)[finite]
        rotation_norm = np.linalg.norm(rotations, axis=1, keepdims=True)
        rotations = rotations / np.maximum(rotation_norm, 1e-8)

    camera_data = np.load(cameras_path)
    w2c = None
    for key in ("w2c_da3", "w2c_render", "w2c_vipe"):
        if key in camera_data and camera_data[key].ndim == 3 and camera_data[key].shape[0] > 0:
            w2c = camera_data[key][0].astype(np.float64)
            break
    if w2c is None:
        raise ValueError("cameras.npz has no usable w2c camera")

    intrinsics = None
    for key in ("intrinsics_da3", "intrinsics_vipe"):
        if key in camera_data and camera_data[key].ndim == 3 and camera_data[key].shape[0] > 0:
            intrinsics = camera_data[key][0].astype(np.float64)
            break
    if intrinsics is None:
        raise ValueError("cameras.npz has no usable intrinsics")

    cam_points = (w2c[:3, :3] @ points.T + w2c[:3, 3:4]).T
    z = cam_points[:, 2]
    valid_z = z > 1e-5
    safe_z = np.maximum(z, 1e-5)
    base_w = max(1.0, float(intrinsics[0, 2]) * 2.0)
    base_h = max(1.0, float(intrinsics[1, 2]) * 2.0)
    sx = width / base_w
    sy = height / base_h
    u = (intrinsics[0, 0] * cam_points[:, 0] / safe_z + intrinsics[0, 2]) * sx
    v = (intrinsics[1, 1] * cam_points[:, 1] / safe_z + intrinsics[1, 2]) * sy
    in_frame = valid_z & (u >= 0.0) & (u < width) & (v >= 0.0) & (v < height)
    alpha_finite = np.isfinite(opacities)
    radius_finite = np.isfinite(world_radii) & (world_radii > 0.0)
    visible = in_frame & alpha_finite & radius_finite & (opacities >= 0.02)
    if np.count_nonzero(visible) < 500:
        visible = in_frame & alpha_finite & radius_finite
    idx = np.where(visible)[0]
    if idx.size == 0:
        raise ValueError("no GS points project into the source camera")

    fx = float(intrinsics[0, 0]) * sx
    fy = float(intrinsics[1, 1]) * sy
    focal_px = (abs(fx) + abs(fy)) * 0.5
    pixel_radii = np.clip(world_radii * focal_px / safe_z * 2.6, 0.9, 7.0)
    ellipse_major = pixel_radii
    ellipse_minor = np.maximum(pixel_radii * 0.82, 0.75)
    ellipse_theta = np.zeros_like(pixel_radii)

    if rotations is not None and scale_world is not None and idx.size > 0:
        q = rotations[idx].astype(np.float32)
        wq, xq, yq, zq = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        rot_mats = np.empty((idx.size, 3, 3), dtype=np.float32)
        rot_mats[:, 0, 0] = 1.0 - 2.0 * (yq * yq + zq * zq)
        rot_mats[:, 0, 1] = 2.0 * (xq * yq - wq * zq)
        rot_mats[:, 0, 2] = 2.0 * (xq * zq + wq * yq)
        rot_mats[:, 1, 0] = 2.0 * (xq * yq + wq * zq)
        rot_mats[:, 1, 1] = 1.0 - 2.0 * (xq * xq + zq * zq)
        rot_mats[:, 1, 2] = 2.0 * (yq * zq - wq * xq)
        rot_mats[:, 2, 0] = 2.0 * (xq * zq - wq * yq)
        rot_mats[:, 2, 1] = 2.0 * (yq * zq + wq * xq)
        rot_mats[:, 2, 2] = 1.0 - 2.0 * (xq * xq + yq * yq)

        axes_world = rot_mats * scale_world[idx, None, :]
        axes_cam = np.einsum("ab,nbc->nac", w2c[:3, :3].astype(np.float32), axes_world)
        x_cam = cam_points[idx, 0].astype(np.float32)
        y_cam = cam_points[idx, 1].astype(np.float32)
        z_cam = np.maximum(z[idx].astype(np.float32), 1e-5)
        inv_z2 = 1.0 / (z_cam * z_cam)
        du = fx * (axes_cam[:, 0, :] * z_cam[:, None] - x_cam[:, None] * axes_cam[:, 2, :]) * inv_z2[:, None]
        dv = fy * (axes_cam[:, 1, :] * z_cam[:, None] - y_cam[:, None] * axes_cam[:, 2, :]) * inv_z2[:, None]
        cov_xx = np.sum(du * du, axis=1) + 0.08
        cov_xy = np.sum(du * dv, axis=1)
        cov_yy = np.sum(dv * dv, axis=1) + 0.08
        trace = cov_xx + cov_yy
        delta = np.sqrt(np.maximum((cov_xx - cov_yy) * (cov_xx - cov_yy) + 4.0 * cov_xy * cov_xy, 0.0))
        lambda_major = np.maximum((trace + delta) * 0.5, 0.08)
        lambda_minor = np.maximum((trace - delta) * 0.5, 0.08)
        major = np.clip(np.sqrt(lambda_major) * 2.7, 0.9, 8.0)
        minor = np.clip(np.sqrt(lambda_minor) * 2.7, 0.65, major)
        theta = 0.5 * np.arctan2(2.0 * cov_xy, cov_xx - cov_yy)
        ellipse_major[idx] = major
        ellipse_minor[idx] = minor
        ellipse_theta[idx] = theta

    order = idx[np.argsort(z[idx])[::-1]]
    alpha_boost = float(min(max(stride, 1), 10))
    alpha_values = 1.0 - np.power(np.clip(1.0 - opacities, 0.0, 1.0), alpha_boost)

    image = np.zeros((height, width, 3), dtype=np.float32)
    image[:, :] = np.array([17.0, 23.0, 21.0], dtype=np.float32)
    kernels: dict[tuple[int, int, int], np.ndarray] = {}

    for i in order:
        x = int(u[i] + 0.5)
        y = int(v[i] + 0.5)
        if x < 0 or x >= width or y < 0 or y >= height:
            continue
        alpha = float(min(max(alpha_values[i] * 0.72, 0.0), 0.94))
        if alpha < 0.006:
            continue
        color = colors[i]
        major = float(ellipse_major[i])
        minor = float(min(ellipse_minor[i], major))
        if major <= 1.05:
            image[y, x] = image[y, x] * (1.0 - alpha) + color * alpha
            continue
        angle_bin = int(round(((float(ellipse_theta[i]) + math.pi) % math.pi) / (math.pi / 18.0))) % 18
        major_bin = max(2, min(16, int(round(major * 2.0))))
        minor_bin = max(1, min(major_bin, int(round(minor * 2.0))))
        key = (major_bin, minor_bin, angle_bin)
        kernel = kernels.get(key)
        if kernel is None:
            major_q = major_bin / 2.0
            minor_q = max(0.5, minor_bin / 2.0)
            radius = int(math.ceil(major_q))
            yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
            theta = angle_bin * (math.pi / 18.0)
            c = math.cos(theta)
            s = math.sin(theta)
            x_rot = xx.astype(np.float32) * c + yy.astype(np.float32) * s
            y_rot = -xx.astype(np.float32) * s + yy.astype(np.float32) * c
            d2 = (x_rot / major_q) ** 2 + (y_rot / minor_q) ** 2
            kernel = np.where(d2 <= 1.0, np.exp(-2.0 * d2), 0.0).astype(np.float32)
            kernels[key] = kernel
        radius = kernel.shape[0] // 2
        y0 = max(0, y - radius)
        y1 = min(height, y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(width, x + radius + 1)
        ky0 = y0 - (y - radius)
        ky1 = ky0 + (y1 - y0)
        kx0 = x0 - (x - radius)
        kx1 = kx0 + (x1 - x0)
        a = (alpha * kernel[ky0:ky1, kx0:kx1])[..., None]
        image[y0:y1, x0:x1] = image[y0:y1, x0:x1] * (1.0 - a) + color * a

    png_image = np.clip(image * 1.03, 0.0, 255.0).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(png_image, mode="RGB").save(buffer, format="PNG", optimize=True)
    png = buffer.getvalue()
    with GS_RASTER_CACHE_LOCK:
        if len(GS_RASTER_CACHE) >= 6:
            GS_RASTER_CACHE.pop(next(iter(GS_RASTER_CACHE)))
        GS_RASTER_CACHE[cache_key] = png
    return png


def _ply_preview(path: Path, max_points: int = 360000) -> dict[str, Any]:
    with path.open("rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("invalid PLY: missing end_header")
            text = line.decode("ascii", errors="replace").strip()
            header_lines.append(text)
            if text == "end_header":
                break
        data_start = f.tell()

    fmt = "ascii"
    vertex_count = 0
    vertex_props: list[tuple[str, str]] = []
    in_vertex = False
    for line in header_lines:
        parts = line.split()
        if not parts:
            continue
        if parts[:2] == ["format", "binary_little_endian"]:
            fmt = "binary_little_endian"
        elif parts[:2] == ["format", "ascii"]:
            fmt = "ascii"
        elif parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                vertex_count = int(parts[2])
        elif parts[0] == "property" and in_vertex and len(parts) >= 3 and parts[1] != "list":
            vertex_props.append((parts[2], parts[1]))

    if vertex_count <= 0:
        raise ValueError("PLY has no vertices")

    stride = max(1, math.ceil(vertex_count / max_points))
    colors = None
    opacities = None
    radii = None
    scales = None
    rotations = None
    is_gaussian = False
    if fmt == "binary_little_endian":
        dtype = np.dtype([(name, "<" + PLY_DTYPES.get(kind, "f4")) for name, kind in vertex_props])
        arr = np.fromfile(path, dtype=dtype, count=vertex_count, offset=data_start)
        arr = arr[::stride]
        names = arr.dtype.names or ()
        points = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32)
        if all(name in names for name in ("red", "green", "blue")):
            colors = np.stack([arr["red"], arr["green"], arr["blue"]], axis=1).astype(np.uint8)
        elif all(name in names for name in ("f_dc_0", "f_dc_1", "f_dc_2")):
            sh0 = 0.28209479177387814
            fdc = np.stack([arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"]], axis=1).astype(np.float32)
            colors = np.clip((0.5 + sh0 * fdc) * 255.0 + 0.5, 0, 255).astype(np.uint8)
        is_gaussian = "opacity" in names and all(f"scale_{idx}" in names for idx in range(3))
        if is_gaussian:
            opacity_raw = arr["opacity"].astype(np.float32)
            opacities = (1.0 / (1.0 + np.exp(-np.clip(opacity_raw, -20.0, 20.0)))).astype(np.float32)
            scale_raw = np.stack([arr[f"scale_{idx}"] for idx in range(3)], axis=1).astype(np.float32)
            scale_world = np.exp(np.clip(scale_raw, -12.0, 4.0)).astype(np.float32)
            scales = scale_world.astype(np.float32)
            radii = np.mean(scale_world, axis=1).astype(np.float32)
            if all(f"rot_{idx}" in names for idx in range(4)):
                rotations = np.stack([arr[f"rot_{idx}"] for idx in range(4)], axis=1).astype(np.float32)
                rotation_norm = np.linalg.norm(rotations, axis=1, keepdims=True)
                rotations = rotations / np.maximum(rotation_norm, 1e-8)
    elif fmt == "ascii":
        prop_names = [name for name, _kind in vertex_props]
        xyz_idx = [prop_names.index(axis) for axis in ("x", "y", "z")]
        rgb_idx = [prop_names.index(axis) for axis in ("red", "green", "blue")] if all(axis in prop_names for axis in ("red", "green", "blue")) else None
        sh_idx = [prop_names.index(axis) for axis in ("f_dc_0", "f_dc_1", "f_dc_2")] if all(axis in prop_names for axis in ("f_dc_0", "f_dc_1", "f_dc_2")) else None
        opacity_idx = prop_names.index("opacity") if "opacity" in prop_names else None
        scale_idx = [prop_names.index(f"scale_{idx}") for idx in range(3)] if all(f"scale_{idx}" in prop_names for idx in range(3)) else None
        rotation_idx = [prop_names.index(f"rot_{idx}") for idx in range(4)] if all(f"rot_{idx}" in prop_names for idx in range(4)) else None
        rows = []
        color_rows = []
        opacity_rows = []
        radius_rows = []
        scale_rows = []
        rotation_rows = []
        with path.open("r", encoding="ascii", errors="replace") as f:
            for line in f:
                if line.strip() == "end_header":
                    break
            for idx, line in enumerate(f):
                if idx % stride:
                    continue
                parts = line.split()
                if len(parts) > max(xyz_idx):
                    rows.append([float(parts[i]) for i in xyz_idx])
                    if rgb_idx and len(parts) > max(rgb_idx):
                        color_rows.append([int(float(parts[i])) for i in rgb_idx])
                    elif sh_idx and len(parts) > max(sh_idx):
                        sh0 = 0.28209479177387814
                        color_rows.append([max(0, min(255, int((0.5 + sh0 * float(parts[i])) * 255.0 + 0.5))) for i in sh_idx])
                    if opacity_idx is not None and len(parts) > opacity_idx:
                        opacity_value = float(parts[opacity_idx])
                        opacity_rows.append(1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, opacity_value)))))
                    if scale_idx and len(parts) > max(scale_idx):
                        scale_values = [math.exp(max(-12.0, min(4.0, float(parts[i])))) for i in scale_idx]
                        scale_rows.append(scale_values)
                        radius_rows.append(sum(scale_values) / 3.0)
                    if rotation_idx and len(parts) > max(rotation_idx):
                        quat = [float(parts[i]) for i in rotation_idx]
                        norm = math.sqrt(sum(v * v for v in quat))
                        if norm > 1e-8:
                            rotation_rows.append([v / norm for v in quat])
                        else:
                            rotation_rows.append([1.0, 0.0, 0.0, 0.0])
        points = np.array(rows, dtype=np.float32)
        if color_rows and len(color_rows) == len(rows):
            colors = np.array(color_rows, dtype=np.uint8)
        is_gaussian = opacity_idx is not None and scale_idx is not None
        if is_gaussian and len(opacity_rows) == len(rows) and len(radius_rows) == len(rows):
            opacities = np.array(opacity_rows, dtype=np.float32)
            radii = np.array(radius_rows, dtype=np.float32)
            if len(scale_rows) == len(rows):
                scales = np.array(scale_rows, dtype=np.float32)
            if len(rotation_rows) == len(rows):
                rotations = np.array(rotation_rows, dtype=np.float32)
    else:
        raise ValueError(f"unsupported PLY format: {fmt}")

    if points.size == 0:
        raise ValueError("no preview points sampled")
    finite = np.all(np.isfinite(points), axis=1)
    points = points[finite]
    if colors is not None:
        colors = colors[finite]
    if scales is not None:
        scales = scales[finite]
    if rotations is not None:
        rotations = rotations[finite]
    if is_gaussian and opacities is not None and radii is not None:
        opacities = opacities[finite]
        radii = radii[finite]
        gaussian_finite = np.isfinite(opacities) & np.isfinite(radii) & (radii > 0)
        if np.count_nonzero(gaussian_finite) > 0:
            points = points[gaussian_finite]
            if colors is not None:
                colors = colors[gaussian_finite]
            if scales is not None:
                scales = scales[gaussian_finite]
            if rotations is not None:
                rotations = rotations[gaussian_finite]
            opacities = opacities[gaussian_finite]
            radii = radii[gaussian_finite]
        if opacities.size:
            visible = opacities >= 0.025
            if np.count_nonzero(visible) >= min(1000, max(1, opacities.size // 10)):
                points = points[visible]
                if colors is not None:
                    colors = colors[visible]
                if scales is not None:
                    scales = scales[visible]
                if rotations is not None:
                    rotations = rotations[visible]
                opacities = opacities[visible]
                radii = radii[visible]
    if points.size == 0:
        raise ValueError("no finite preview points sampled")

    if is_gaussian and points.shape[0] >= 1000:
        mins = np.percentile(points, 0.5, axis=0)
        maxs = np.percentile(points, 99.5, axis=0)
    else:
        mins = np.min(points, axis=0)
        maxs = np.max(points, axis=0)
    center = (mins + maxs) * 0.5
    norm_scale = float(np.max(maxs - mins))
    if norm_scale <= 1e-8:
        norm_scale = 1.0
    points = (points - center) / norm_scale * 2.2
    view_hint = _gs_camera_view_hint(path, center, norm_scale) if is_gaussian else None
    if is_gaussian and opacities is not None and radii is not None:
        if scales is not None:
            scales = scales / norm_scale * 2.2
            radii = np.mean(scales, axis=1).astype(np.float32)
        else:
            radii = radii / norm_scale * 2.2
        if radii.size:
            radius_hi = float(np.percentile(radii, 99.5))
            if math.isfinite(radius_hi) and radius_hi > 0:
                old_radii = np.maximum(radii, 1e-8)
                clipped_radii = np.clip(radii, radius_hi * 0.05, radius_hi)
                if scales is not None:
                    scales = scales * (clipped_radii / old_radii)[:, None]
                radii = clipped_radii
        alpha_boost = float(min(max(stride, 1), 8))
        opacities = 1.0 - np.power(np.clip(1.0 - opacities, 0.0, 1.0), alpha_boost)
        opacities = np.clip(opacities * 0.78, 0.015, 0.96).astype(np.float32)
        if view_hint is not None and points.shape[0] > 1:
            eye = np.array(view_hint["eye"], dtype=np.float32)
            target = np.array(view_hint["target"], dtype=np.float32)
            forward = target - eye
            forward_norm = np.linalg.norm(forward)
            if forward_norm > 1e-8:
                forward = forward / forward_norm
                points_for_depth = points.copy()
                points_for_depth[:, 1] *= -1.0
                depths = (points_for_depth - eye[None, :]) @ forward
                order = np.argsort(depths)[::-1]
                points = points[order]
                if colors is not None:
                    colors = colors[order]
                if scales is not None:
                    scales = scales[order]
                if rotations is not None:
                    rotations = rotations[order]
                opacities = opacities[order]
                radii = radii[order]

    payload = {
        "count": int(vertex_count),
        "sampled": int(points.shape[0]),
        "points": points[:, :3].round(5).tolist(),
        "kind": "gaussian" if is_gaussian else "point_cloud",
    }
    if view_hint is not None:
        payload["view_hint"] = view_hint
    if colors is not None:
        payload["colors"] = colors[:, :3].tolist()
    if is_gaussian and opacities is not None and radii is not None:
        payload["opacities"] = opacities.round(4).tolist()
        payload["radii"] = radii.round(6).tolist()
        if scales is not None:
            payload["scales"] = scales.round(6).tolist()
        if rotations is not None:
            payload["rotations"] = rotations.round(6).tolist()
    return payload

def _index_html() -> str:
    preset_options = "\n".join(
        f'<option value="{html.escape(key)}">{html.escape(label)}</option>' for key, label in PRESET_TRAJECTORIES.items()
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Lyra-2 Demo UI</title>
  <style>
    :root {{
      --bg: #f5f7f6;
      --panel: #ffffff;
      --ink: #18211f;
      --muted: #63706b;
      --line: #d8e0dc;
      --accent: #247c62;
      --accent-hover: #1d6350;
      --accent-2: #b55235;
      --soft: #eaf2ee;
      --warn: #8a5a00;
      --bad: #a83232;
      --success: #2d7a5f;
      --info: #4a7c9d;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }}
    header {{
      height: 58px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 22px;
      border-bottom: 1px solid var(--line);
      background: rgba(255,255,255,0.92);
      position: sticky;
      top: 0;
      z-index: 5;
    }}
    h1 {{
      font-size: 18px;
      margin: 0;
      font-weight: 700;
    }}
    .status-pill {{
      min-width: 108px;
      border: 1px solid var(--line);
      background: var(--soft);
      color: var(--muted);
      border-radius: 999px;
      padding: 7px 12px;
      text-align: center;
      font-size: 13px;
      font-weight: 600;
      transition: all 0.2s ease;
    }}
    .status-pill.success {{
      background: #e8f5f0;
      color: var(--success);
      border-color: #b8dccf;
    }}
    .status-pill.error {{
      background: #fdeaea;
      color: var(--bad);
      border-color: #f5c6c6;
    }}
    .status-pill.working {{
      background: #fff9e6;
      color: var(--warn);
      border-color: #f5e5b8;
      animation: pulse 2s ease-in-out infinite;
    }}
    @keyframes pulse {{
      0%, 100% {{ opacity: 1; }}
      50% {{ opacity: 0.7; }}
    }}
    .spinner {{
      display: inline-block;
      width: 14px;
      height: 14px;
      border: 2px solid rgba(255,255,255,0.3);
      border-top-color: white;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
      margin-right: 6px;
      vertical-align: middle;
    }}
    @keyframes spin {{
      to {{ transform: rotate(360deg); }}
    }}
    .worker-strip {{
      display: grid;
      grid-template-columns: 1fr auto;
      align-items: center;
      gap: 10px;
      margin-top: 12px;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfdfc;
    }}
    .worker-actions {{
      display: flex;
      flex-direction: column;
      align-items: stretch;
      justify-content: center;
      gap: 8px;
      min-width: 142px;
    }}
    .worker-actions button {{ width: 100%; white-space: nowrap; }}
    .worker-meta {{
      display: grid;
      gap: 4px;
      min-width: 0;
      font-size: 13px;
      font-weight: 700;
    }}
    .worker-state {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 500;
      overflow-wrap: anywhere;
    }}
    .worker-state.ready {{ color: var(--accent); }}
    .worker-state.starting, .worker-state.loading, .worker-state.busy, .worker-state.offloading, .worker-state.offloaded, .worker-state.moving_gpu {{ color: var(--warn); }}
    .worker-state.failed, .worker-state.exited {{ color: var(--bad); }}
    .video-picker {{
      position: absolute;
      top: 10px;
      right: 10px;
      z-index: 3;
      max-width: calc(100% - 132px);
    }}
    .video-picker select {{
      margin: 0;
      width: min(360px, 34vw);
      min-height: 30px;
      padding: 5px 8px;
      border-radius: 6px;
      background: rgba(255,255,255,0.92);
      font-size: 12px;
    }}
    .scene-pickers {{
      position: absolute;
      top: 42px;
      left: 10px;
      right: 10px;
      z-index: 3;
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      max-width: calc(100% - 20px);
      pointer-events: none;
    }}
    .scene-pickers select {{
      margin: 0;
      width: min(230px, calc(33vw - 18px));
      min-width: 136px;
      min-height: 30px;
      padding: 5px 8px;
      border-radius: 6px;
      background: rgba(255,255,255,0.92);
      font-size: 12px;
      pointer-events: auto;
    }}
    main {{
      display: grid;
      grid-template-columns: minmax(300px, 360px) minmax(680px, 1fr);
      gap: 14px;
      padding: 14px;
      min-height: calc(100vh - 58px);
    }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      min-width: 0;
    }}
    .panel {{
      padding: 14px;
      overflow: auto;
      max-height: calc(100vh - 86px);
    }}
    h2 {{
      font-size: 13px;
      text-transform: uppercase;
      color: var(--muted);
      margin: 4px 0 12px;
      letter-spacing: 0;
    }}
    label {{
      display: block;
      font-size: 13px;
      color: var(--muted);
      margin: 12px 0 6px;
    }}
    .strength-label span {{
      display: block;
      margin-top: 4px;
      font-size: 12px;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }}
    input[type="text"], input[type="number"], textarea, select {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 10px;
      color: var(--ink);
      background: #fff;
      font: inherit;
      font-size: 14px;
      transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }}
    input[type="text"]:focus, input[type="number"]:focus, textarea:focus, select:focus {{
      outline: none;
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(36, 124, 98, 0.1);
    }}
    textarea {{ min-height: 86px; resize: vertical; }}
    input[type="file"] {{
      width: 100%;
      border: 1px dashed #b9c8c1;
      border-radius: 6px;
      padding: 10px;
      background: #fbfdfc;
    }}
    .input-preview-card {{
      height: 170px;
      margin: 12px 0 6px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #0f1513;
      overflow: hidden;
      display: grid;
      place-items: center;
    }}
    .input-preview-card img {{
      width: 100%;
      height: 100%;
      object-fit: contain;
      background: #0f1513;
    }}
    .row {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }}
    .checks {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 10px;
    }}
    .check {{
      display: flex;
      align-items: center;
      gap: 7px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px;
      color: var(--ink);
      font-size: 13px;
      min-height: 38px;
    }}
    button {{
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 6px;
      padding: 9px 11px;
      font: inherit;
      cursor: pointer;
      min-height: 38px;
      transition: all 0.15s ease;
    }}
    button:hover:not(:disabled) {{
      background: var(--soft);
      border-color: var(--accent);
    }}
    button.primary {{
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
      font-weight: 700;
    }}
    button.primary:hover:not(:disabled) {{
      background: var(--accent-hover);
      border-color: var(--accent-hover);
      transform: translateY(-1px);
      box-shadow: 0 2px 8px rgba(36, 124, 98, 0.2);
    }}
    button.secondary {{
      background: #fcf4ef;
      border-color: #e3c9ba;
      color: #74371f;
      font-weight: 700;
    }}
    button.secondary:hover:not(:disabled) {{
      background: #f9ebe1;
      transform: translateY(-1px);
    }}
    button:disabled {{
      opacity: 0.55;
      cursor: not-allowed;
      transform: none !important;
    }}
    .button-row {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 14px;
    }}
    .action-row {{ align-items: start; }}
    .action-cell {{
      display: grid;
      gap: 7px;
      min-width: 0;
    }}
    .action-cell button {{ width: 100%; }}
    .recon-hint {{
      grid-column: 1 / -1;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }}
    .keys {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 7px;
      margin-top: 8px;
    }}
    .keys button {{ font-weight: 700; }}
    .keys .empty {{ visibility: hidden; }}
    .chips {{
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      min-height: 34px;
      margin-top: 8px;
      padding: 7px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfdfc;
    }}
    .chip {{
      border-radius: 999px;
      background: var(--soft);
      color: #29463c;
      padding: 4px 8px;
      font-size: 12px;
    }}
    details {{
      margin-top: 12px;
      border-top: 1px solid var(--line);
      padding-top: 10px;
    }}
    summary {{ cursor: pointer; color: var(--muted); font-size: 13px; }}
    .media-grid {{
      display: grid;
      grid-template-rows: 240px minmax(360px, 1fr);
      gap: 14px;
      height: calc(100vh - 86px);
      padding: 14px;
    }}
    .history-strip {{
      display: grid;
      grid-template-columns: minmax(180px, 0.85fr) minmax(220px, 1fr) minmax(300px, 1.35fr);
      gap: 10px;
      min-height: 0;
    }}
    .history-card {{
      min-height: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfdfc;
      padding: 10px;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      overflow: hidden;
    }}
    .history-card h2 {{
      margin: 0 0 8px;
    }}
    .history-card .path-list {{
      min-height: 0;
      margin: 0;
      overflow: auto;
    }}
    .history-card .path-list a {{
      padding: 7px;
      font-size: 12px;
    }}
    .history-card pre {{
      min-height: 0;
      height: 100%;
      padding: 8px;
      overflow: auto;
    }}
    .media-slot {{
      min-height: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #0f1513;
      overflow: hidden;
      position: relative;
      display: grid;
      place-items: center;
    }}
    .media-slot img, .media-slot video {{
      max-width: 100%;
      max-height: 100%;
      width: 100%;
      height: 100%;
      object-fit: contain;
      background: #0f1513;
    }}
    .media-label {{
      position: absolute;
      top: 10px;
      left: 10px;
      background: rgba(255,255,255,0.88);
      color: var(--ink);
      border-radius: 999px;
      padding: 5px 9px;
      font-size: 12px;
      border: 1px solid rgba(0,0,0,0.08);
    }}
    .split {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
      min-height: 0;
    }}
    canvas {{
      width: 100%;
      height: 100%;
      display: block;
      background: #111715;
    }}
    #pointCanvas {{
      cursor: grab;
      touch-action: none;
    }}
    .scene-raster {{
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      object-fit: contain;
      background: #111715;
      pointer-events: none;
      z-index: 1;
    }}
    #pointCanvas.dragging {{ cursor: grabbing; }}
    #pointCanvas.hidden, #sceneVideoPreview.hidden, #gsRasterPreview.hidden {{ display: none; }}
    .scene-empty, .scene-info {{
      position: absolute;
      left: 12px;
      z-index: 2;
      border: 1px solid rgba(255,255,255,0.10);
      border-radius: 6px;
      background: rgba(14,20,18,0.74);
      color: #dfeae4;
      font-size: 12px;
      line-height: 1.35;
      pointer-events: none;
    }}
    .scene-empty {{
      top: 78px;
      max-width: calc(100% - 24px);
      padding: 8px 10px;
    }}
    .scene-info {{
      bottom: 12px;
      max-width: calc(100% - 24px);
      padding: 6px 8px;
      color: #b9cac2;
    }}
    .scene-empty.hidden, .scene-info.hidden {{ display: none; }}
    .scene-toolbar {{
      position: absolute;
      top: 10px;
      right: 10px;
      display: inline-flex;
      gap: 2px;
      padding: 3px;
      background: rgba(255,255,255,0.90);
      border: 1px solid rgba(0,0,0,0.10);
      border-radius: 6px;
      z-index: 2;
    }}
    .scene-toolbar button {{
      border: 0;
      border-radius: 4px;
      padding: 5px 9px;
      min-width: 48px;
      background: transparent;
      color: var(--muted);
      font-size: 12px;
      cursor: pointer;
    }}
    .scene-toolbar button.active {{
      background: var(--accent);
      color: white;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      margin: 0;
      padding: 10px;
      background: #101614;
      color: #e6eee9;
      border-radius: 6px;
      min-height: 260px;
      font-size: 12px;
      line-height: 1.45;
      overflow: auto;
    }}
    .path-list {{
      display: grid;
      gap: 8px;
      margin-bottom: 10px;
    }}
    .path-list a {{
      display: block;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      color: var(--accent);
      text-decoration: none;
      overflow-wrap: anywhere;
      font-size: 13px;
      background: #fbfdfc;
    }}
    .small {{
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
    }}
    .hidden {{ display: none; }}
    .toast {{
      position: fixed;
      bottom: 20px;
      right: 20px;
      background: var(--ink);
      color: white;
      padding: 12px 18px;
      border-radius: 8px;
      box-shadow: 0 4px 16px rgba(0,0,0,0.2);
      z-index: 1000;
      animation: slideIn 0.3s ease;
      max-width: 400px;
      font-size: 14px;
    }}
    .toast.success {{ background: var(--success); }}
    .toast.error {{ background: var(--bad); }}
    .toast.info {{ background: var(--info); }}
    @keyframes slideIn {{
      from {{ transform: translateX(400px); opacity: 0; }}
      to {{ transform: translateX(0); opacity: 1; }}
    }}
    .kbd {{
      display: inline-block;
      padding: 2px 6px;
      background: var(--soft);
      border: 1px solid var(--line);
      border-radius: 4px;
      font-family: monospace;
      font-size: 11px;
      font-weight: 600;
      color: var(--muted);
    }}
    .help-icon {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 18px;
      height: 18px;
      border-radius: 50%;
      background: var(--soft);
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      cursor: help;
      margin-left: 6px;
    }}
    @media (max-width: 1120px) {{
      main {{ grid-template-columns: 1fr; }}
      .panel, .media-grid {{ max-height: none; height: auto; }}
      .media-grid {{ grid-template-rows: auto 420px; }}
      .history-strip {{ grid-template-columns: 1fr; grid-auto-rows: 210px; }}
    }}
    @media (max-width: 760px) {{
      .split {{ grid-template-columns: 1fr; }}
      .media-grid {{ grid-template-rows: auto auto; }}
      .media-slot {{ min-height: 360px; }}
    }}
    .modal {{
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0,0,0,0.5);
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 1000;
      padding: 20px;
    }}
    .modal.active {{ display: flex; }}
    .modal-content {{
      background: white;
      border-radius: 12px;
      padding: 24px;
      max-width: 600px;
      max-height: 80vh;
      overflow-y: auto;
      box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }}
    .modal-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 20px;
      padding-bottom: 12px;
      border-bottom: 2px solid var(--line);
    }}
    .modal-header h3 {{
      margin: 0;
      font-size: 20px;
      color: var(--ink);
    }}
    .modal-close {{
      background: none;
      border: none;
      font-size: 24px;
      color: var(--muted);
      cursor: pointer;
      padding: 0;
      width: 32px;
      height: 32px;
      display: flex;
      align-items: center;
      justify-content: center;
      border-radius: 4px;
      min-height: auto;
    }}
    .modal-close:hover {{
      background: var(--soft);
      color: var(--ink);
    }}
    .shortcut-list {{
      display: grid;
      gap: 12px;
    }}
    .shortcut-item {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 10px;
      background: var(--soft);
      border-radius: 6px;
    }}
    .shortcut-key {{
      background: white;
      border: 1px solid var(--line);
      padding: 4px 10px;
      border-radius: 4px;
      font-family: monospace;
      font-weight: 700;
      font-size: 13px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Lyra-2 Demo UI</h1>
    <div style="display: flex; align-items: center; gap: 12px;">
      <button id="helpBtn" type="button" style="min-height: 32px; padding: 6px 12px; font-size: 13px;" title="Show keyboard shortcuts">
        <span style="font-weight: 700;">?</span> Help
      </button>
      <div class="status-pill" id="statusPill">idle</div>
    </div>
  </header>
  <main>
    <section class="panel">
      <h2>Input</h2>
      <label for="imageInput">Image <span class="help-icon" title="Upload an image or select from samples below">?</span></label>
      <input id="imageInput" type="file" accept="image/png,image/jpeg" />
      <label for="sampleSelect">Sample</label>
      <select id="sampleSelect"></select>
      <div class="input-preview-card">
        <img id="inputPreview" alt="" />
      </div>
      <label for="promptInput">Prompt <span class="help-icon" title="Describe the scene you want to generate">?</span></label>
      <textarea id="promptInput">{html.escape(DEFAULT_PROMPT)}</textarea>

      <div class="row">
        <div>
          <label for="durationInput">Seconds</label>
          <input id="durationInput" type="number" min="5" max="120" step="5" value="5" />
        </div>
        <div>
          <label for="fpsInput">FPS</label>
          <input id="fpsInput" type="number" min="1" max="30" step="1" value="16" />
        </div>
      </div>

      <h2 style="margin-top:18px;">Camera</h2>
      <label for="trajectoryMode">Source</label>
      <select id="trajectoryMode">
        <option value="preset">Preset trajectory</option>
        <option value="keyboard">W/S/A/D + arrows</option>
      </select>
      <div id="presetBlock">
        <label for="presetSelect">Preset</label>
        <select id="presetSelect">{preset_options}</select>
      </div>
      <div id="keyboardBlock" class="hidden">
        <div class="keys">
          <span class="empty"></span><button data-op="up">Up</button><span class="empty"></span>
          <button data-op="left">Left</button><button data-op="w">W</button><button data-op="right">Right</button>
          <button data-op="a">A</button><button data-op="s">S</button><button data-op="d">D</button>
          <span class="empty"></span><button data-op="down">Down</button><span class="empty"></span>
        </div>
        <div class="chips" id="opChips"></div>
        <div class="button-row">
          <button id="clearOps" type="button">Clear</button>
          <button id="sampleOps" type="button">Sample Path</button>
        </div>
      </div>
      <label for="strengthInput" class="strength-label">Strength
        <span>Per key press: W/S/A/D move 0.28 x Strength; Up/Down/Left/Right rotate 12 deg x Strength. Up/Down pitch is capped at +/-45 deg.</span>
      </label>
      <input id="strengthInput" type="number" min="0.1" max="3" step="0.1" value="1.0" />

      <div class="checks">
        <label class="check"><input id="dmdInput" type="checkbox" checked /> DMD fast</label>
        <label class="check"><input id="step1CloudInput" type="checkbox" checked /> Point cloud</label>
        <label class="check"><input id="autoReconInput" type="checkbox" /> Auto recon</label>
        <label class="check"><input id="offloadReconInput" type="checkbox" checked /> Offload worker for recon</label>
        <label class="check"><input id="arContinueInput" type="checkbox" checked /> AR continuation</label>
      </div>

      <div class="worker-strip">
        <div class="worker-meta">
          <span>Generation mode</span>
          <span class="worker-state" id="sceneModeText">New scene</span>
        </div>
        <button id="resetSceneBtn" type="button">Reset Scene</button>
      </div>

      <div class="worker-strip">
        <div class="worker-meta">
          <span>Model preload</span>
          <span class="worker-state" id="workerStatus">stopped</span>
        </div>
        <div class="worker-actions">
          <button id="preloadBtn" type="button">Preload Models</button>
          <button id="releaseBtn" type="button" disabled>Release Models</button>
        </div>
      </div>

      <details>
        <summary>Advanced</summary>
        <label for="experimentInput">Experiment</label>
        <input id="experimentInput" type="text" value="lyra2" />
        <label for="checkpointInput">Checkpoint</label>
        <input id="checkpointInput" type="text" value="checkpoints/model" />
        <div class="row">
          <div>
            <label for="resolutionInput">Resolution</label>
            <input id="resolutionInput" type="text" value="480,832" />
          </div>
          <div>
            <label for="poseScaleInput">Pose scale</label>
            <input id="poseScaleInput" type="number" min="0.1" max="5" step="0.1" value="1.1" />
          </div>
        </div>
        <div class="row">
          <div>
            <label for="seedInput">Seed</label>
            <input id="seedInput" type="number" value="1" />
          </div>
          <div>
            <label for="stepsInput">Steps</label>
            <input id="stepsInput" type="number" min="1" max="80" value="35" />
          </div>
        </div>
        <label for="sparsePointsInput">Point cloud points</label>
        <input id="sparsePointsInput" type="number" min="1000" max="1000000" step="10000" value="200000" />
        <div class="checks">
          <label class="check"><input id="mogeInput" type="checkbox" checked /> MoGe scale</label>
          <label class="check"><input id="forceReconInput" type="checkbox" /> Force recon</label>
        </div>
      </details>

      <div class="button-row action-row">
        <div class="action-cell">
          <button class="primary" id="generateBtn" type="button">
            <span id="generateBtnText">Generate AR Video</span>
          </button>
        </div>
        <div class="action-cell">
          <button class="secondary" id="reconBtn" type="button">
            <span id="reconBtnText">Reconstruct Scene</span>
          </button>
        </div>
        <div class="recon-hint">Reminder: Select the target segment or Full scene in Generated video before reconstruction. Auto recon reconstructs the current segment automatically; for Full scene, select it manually and click Reconstruct Scene.</div>
      </div>
      <button id="cancelBtn" type="button" style="width:100%;margin-top:10px;" disabled>Cancel Current Job</button>
    </section>

    <section class="media-grid">
      <div class="history-strip" aria-label="Run history">
        <div class="history-card">
          <h2>Outputs</h2>
          <div class="path-list" id="paths"></div>
        </div>
        <div class="history-card">
          <h2>Command</h2>
          <pre id="commandBox"></pre>
        </div>
        <div class="history-card">
          <h2>Log</h2>
          <pre id="logBox"></pre>
        </div>
      </div>
      <div class="split">
        <div class="media-slot">
          <span class="media-label">Generated video</span>
          <div class="video-picker">
            <select id="videoSelect" disabled><option>No generated video</option></select>
          </div>
          <video id="videoPreview" controls playsinline></video>
        </div>
        <div class="media-slot">
          <span class="media-label">Point cloud / GS scene</span>
          <div class="scene-pickers">
            <select id="pdSelect" disabled><option>No Point cloud</option></select>
            <select id="gsSelect" disabled><option>No GS scene</option></select>
            <select id="gsRenderSelect" disabled><option>No GS_Render</option></select>
          </div>
          <div class="scene-toolbar" role="group" aria-label="Scene view">
            <button type="button" data-scene-view="step1" class="active">PC</button>
            <button type="button" data-scene-view="gs">GS</button>
            <button type="button" data-scene-view="video">GS_Render</button>
          </div>
          <canvas id="pointCanvas"></canvas>
          <img id="gsRasterPreview" class="scene-raster hidden" alt="" />
          <video id="sceneVideoPreview" class="hidden" controls playsinline></video>
          <div class="scene-empty" id="sceneEmpty">Point clouds appear after generation / reconstruction</div>
          <div class="scene-info hidden" id="sceneInfo"></div>
        </div>
      </div>
    </section>
  </main>

  <div class="modal" id="helpModal">
    <div class="modal-content">
      <div class="modal-header">
        <h3>Keyboard Shortcuts & Tips</h3>
        <button class="modal-close" id="closeHelp" type="button">&times;</button>
      </div>
      <div class="shortcut-list">
        <div class="shortcut-item">
          <span>Start generation</span>
          <span class="shortcut-key">G</span>
        </div>
        <div class="shortcut-item">
          <span>Start reconstruction</span>
          <span class="shortcut-key">R</span>
        </div>
        <div class="shortcut-item">
          <span>Cancel current job</span>
          <span class="shortcut-key">Esc</span>
        </div>
        <div class="shortcut-item">
          <span>Save settings</span>
          <span class="shortcut-key">Ctrl+S</span>
        </div>
        <div class="shortcut-item">
          <span>Add camera operation (keyboard mode)</span>
          <span class="shortcut-key">W/A/S/D/Arrows</span>
        </div>
      </div>
      <div style="margin-top: 20px; padding: 12px; background: var(--soft); border-radius: 6px; font-size: 13px; line-height: 1.6;">
        <strong>Tips:</strong>
        <ul style="margin: 8px 0 0 0; padding-left: 20px;">
          <li>Settings are auto-saved as you change them</li>
          <li>Use "Preload Models" for faster generation</li>
          <li>Enable "Auto recon" to automatically reconstruct after generation</li>
          <li>Drag on the point cloud viewer to rotate the scene</li>
          <li>Use mouse wheel to zoom in/out on point clouds</li>
        </ul>
      </div>
    </div>
  </div>

<script>
const samples = [];
let currentJobId = null;
let currentGeneratedJobId = null;
let currentReconJobId = null;
let pollTimer = null;
let workerPollTimer = null;
let lastWorkerStatus = {{ status: 'stopped', running: false }};
let activeLogSource = 'idle';
let currentScene = null;
let selectedVideoPath = '';
let selectedPdPath = '';
let selectedGsPath = '';
let selectedGsRenderPath = '';
let pdUserSelected = false;
let gsUserSelected = false;
let gsRenderUserSelected = false;
let reconRestoreAfterJob = false;
let operations = [];
let pointClouds = {{ step1: null, gs: null }};
let sceneWebgl = null;
let sceneViewMode = 'step1';
let sceneTrajectoryUrl = '';
let sceneYaw = 0;
let scenePitch = -0.18;
let sceneZoom = 1.0;
let sceneOrbitTarget = [0, 0, 0];
let sceneOrbitDistance = 4.4;
let sceneOrbitFov = Math.PI / 4;
let sceneOrbitPath = '';
let sceneDragging = false;
let sceneUserControlled = false;
let scenePointerMode = 'orbit';
let sceneLastPointer = {{ x: 0, y: 0 }};

const $ = (id) => document.getElementById(id);

function showToast(message, type='info', duration=3000) {{
  const toast = document.createElement('div');
  toast.className = `toast ${{type}}`;
  toast.textContent = message;
  document.body.appendChild(toast);
  setTimeout(() => {{
    toast.style.animation = 'slideIn 0.3s ease reverse';
    setTimeout(() => toast.remove(), 300);
  }}, duration);
}}

function setStatus(text, bad=false) {{
  const pill = $('statusPill');
  pill.textContent = text;
  pill.className = 'status-pill';
  if (bad) {{
    pill.classList.add('error');
  }} else if (text.includes('running') || text.includes('queued') || text.includes('loading')) {{
    pill.classList.add('working');
  }} else if (text.includes('complete') || text.includes('ready')) {{
    pill.classList.add('success');
  }}
}}

function saveSettings() {{
  const settings = {{
    prompt: $('promptInput').value,
    duration: $('durationInput').value,
    fps: $('fpsInput').value,
    trajectoryMode: $('trajectoryMode').value,
    preset: $('presetSelect').value,
    strength: $('strengthInput').value,
    dmd: $('dmdInput').checked,
    step1Cloud: $('step1CloudInput').checked,
    autoRecon: $('autoReconInput').checked,
    offloadRecon: $('offloadReconInput').checked,
    arContinue: $('arContinueInput').checked,
    experiment: $('experimentInput').value,
    checkpoint: $('checkpointInput').value,
    resolution: $('resolutionInput').value,
    poseScale: $('poseScaleInput').value,
    seed: $('seedInput').value,
    steps: $('stepsInput').value,
    sparsePoints: $('sparsePointsInput').value,
    moge: $('mogeInput').checked,
  }};
  const fd = new FormData();
  fd.append('settings', JSON.stringify(settings));
  fetch('/api/save_settings', {{ method: 'POST', body: fd }})
    .then(() => showToast('Settings saved', 'success', 2000))
    .catch(() => showToast('Failed to save settings', 'error'));
}}

function loadSettings(settings) {{
  if (!settings) return;
  if (settings.prompt) $('promptInput').value = settings.prompt;
  if (settings.duration) $('durationInput').value = settings.duration;
  if (settings.fps) $('fpsInput').value = settings.fps;
  if (settings.trajectoryMode) $('trajectoryMode').value = settings.trajectoryMode;
  if (settings.preset) $('presetSelect').value = settings.preset;
  if (settings.strength) $('strengthInput').value = settings.strength;
  if (settings.dmd !== undefined) $('dmdInput').checked = settings.dmd;
  if (settings.step1Cloud !== undefined) $('step1CloudInput').checked = settings.step1Cloud;
  if (settings.autoRecon !== undefined) $('autoReconInput').checked = settings.autoRecon;
  if (settings.offloadRecon !== undefined) $('offloadReconInput').checked = settings.offloadRecon;
  if (settings.arContinue !== undefined) $('arContinueInput').checked = settings.arContinue;
  if (settings.experiment) $('experimentInput').value = settings.experiment;
  if (settings.checkpoint) $('checkpointInput').value = settings.checkpoint;
  if (settings.resolution) $('resolutionInput').value = settings.resolution;
  if (settings.poseScale) $('poseScaleInput').value = settings.poseScale;
  if (settings.seed) $('seedInput').value = settings.seed;
  if (settings.steps) $('stepsInput').value = settings.steps;
  if (settings.sparsePoints) $('sparsePointsInput').value = settings.sparsePoints;
  if (settings.moge !== undefined) $('mogeInput').checked = settings.moge;
  updateMode();
}}

function setBusy(busy) {{
  $('generateBtn').disabled = busy;
  $('reconBtn').disabled = busy && !currentGeneratedJobId;
  $('cancelBtn').disabled = !busy;
  if (busy) {{
    $('preloadBtn').disabled = true;
    $('releaseBtn').disabled = true;
  }} else {{
    renderWorkerStatus(lastWorkerStatus);
  }}
}}

function updateMode() {{
  const mode = $('trajectoryMode').value;
  $('presetBlock').classList.toggle('hidden', mode !== 'preset');
  $('keyboardBlock').classList.toggle('hidden', mode !== 'keyboard');
}}

function renderOps() {{
  const box = $('opChips');
  box.innerHTML = '';
  operations.forEach((op) => {{
    const chip = document.createElement('span');
    chip.className = 'chip';
    chip.textContent = op.toUpperCase();
    box.appendChild(chip);
  }});
}}

function addOp(op) {{
  operations.push(op);
  renderOps();
}}

function fileNameFromPath(path) {{
  return path.split('/').pop();
}}

async function loadSamples() {{
  const res = await fetch('/api/samples');
  const data = await res.json();
  const select = $('sampleSelect');
  select.innerHTML = '<option value="">No sample</option>';
  data.samples.forEach((item) => {{
    samples.push(item);
    const option = document.createElement('option');
    option.value = item.path;
    option.textContent = item.name;
    select.appendChild(option);
  }});
  if (samples.length) {{
    select.value = samples[Math.min(4, samples.length - 1)].path;
    onSampleChange();
  }}
}}

function nextGenerationContinues() {{
  return Boolean($('arContinueInput').checked && currentGeneratedJobId);
}}

function updateSceneMode() {{
  const el = $('sceneModeText');
  if (!el) return;
  if (nextGenerationContinues()) {{
    const count = currentScene && currentScene.segment_count ? Number(currentScene.segment_count) : 1;
    el.textContent = `AR continuation: next segment ${{String(count + 1).padStart(2, '0')}}`;
    el.className = 'worker-state ready';
  }} else {{
    el.textContent = 'New scene generation';
    el.className = 'worker-state';
  }}
}}

function renderVideoSelect(scene) {{
  const select = $('videoSelect');
  if (!select) return '';
  if (scene) currentScene = scene;
  const videos = ((scene && scene.videos) || []).filter((item) => item && item.url);
  const previous = selectedVideoPath || select.value;
  select.innerHTML = '';
  if (!videos.length) {{
    const option = document.createElement('option');
    option.textContent = 'No generated video';
    option.value = '';
    select.appendChild(option);
    select.disabled = true;
    selectedVideoPath = '';
    updateSceneMode();
    return '';
  }}
  videos.forEach((item) => {{
    const option = document.createElement('option');
    option.value = item.path;
    option.dataset.url = item.url;
    option.textContent = item.label;
    select.appendChild(option);
  }});
  const chosen = videos.find((item) => item.path === previous) || videos[0];
  select.value = chosen.path;
  selectedVideoPath = chosen.path;
  select.disabled = false;
  updateSceneMode();
  return chosen.url;
}}

async function renderPdSelect(scene, preserveExisting=false) {{
  const select = $('pdSelect');
  if (!select) return;
  const clouds = ((scene && scene.pd_clouds) || []).filter((item) => item && item.url);
  const previous = selectedPdPath || select.value;
  if (!clouds.length && preserveExisting && previous) {{
    return;
  }}
  select.innerHTML = '';
  if (!clouds.length) {{
    const option = document.createElement('option');
    option.textContent = 'No Point cloud';
    option.value = '';
    select.appendChild(option);
    select.disabled = true;
    selectedPdPath = '';
    pdUserSelected = false;
    disposeSceneCloud(pointClouds.step1);
    pointClouds.step1 = null;
    return;
  }}
  clouds.forEach((item) => {{
    const option = document.createElement('option');
    option.value = item.path;
    option.dataset.url = item.url;
    option.textContent = item.label;
    select.appendChild(option);
  }});
  let chosen = pdUserSelected ? clouds.find((item) => item.path === previous) : null;
  if (!chosen) {{
    chosen = clouds[clouds.length - 1];
    pdUserSelected = false;
  }}
  select.value = chosen.path;
  selectedPdPath = chosen.path;
  select.disabled = false;
  await loadPlyInto('step1', chosen.path);
}}

async function renderGsSelect(scene, preserveExisting=false) {{
  const select = $('gsSelect');
  if (!select) return;
  const clouds = ((scene && scene.gs_clouds) || []).filter((item) => item && item.url);
  const previous = selectedGsPath || select.value;
  if (!clouds.length && preserveExisting && previous) {{
    return;
  }}
  select.innerHTML = '';
  if (!clouds.length) {{
    const option = document.createElement('option');
    option.textContent = 'No GS scene';
    option.value = '';
    select.appendChild(option);
    select.disabled = true;
    selectedGsPath = '';
    gsUserSelected = false;
    disposeSceneCloud(pointClouds.gs);
    pointClouds.gs = null;
    applySceneDisplay();
    return;
  }}
  clouds.forEach((item) => {{
    const option = document.createElement('option');
    option.value = item.path;
    option.dataset.url = item.url;
    option.textContent = item.label;
    select.appendChild(option);
  }});
  let chosen = gsUserSelected ? clouds.find((item) => item.path === previous) : null;
  if (!chosen) chosen = clouds.find((item) => item.path === previous) || clouds[clouds.length - 1];
  if (!gsUserSelected && chosen.path !== previous) gsUserSelected = false;
  select.value = chosen.path;
  selectedGsPath = chosen.path;
  select.disabled = false;
  await loadPlyInto('gs', chosen.path);
}}

function renderGsRenderSelect(scene, preserveExisting=false) {{
  const select = $('gsRenderSelect');
  if (!select) return;
  const renders = ((scene && scene.gs_renders) || []).filter((item) => item && item.url);
  const previous = selectedGsRenderPath || select.value;
  if (!renders.length && preserveExisting && previous) {{
    return;
  }}
  select.innerHTML = '';
  if (!renders.length) {{
    const option = document.createElement('option');
    option.textContent = 'No GS_Render';
    option.value = '';
    select.appendChild(option);
    select.disabled = true;
    selectedGsRenderPath = '';
    gsRenderUserSelected = false;
    sceneTrajectoryUrl = '';
    applySceneDisplay();
    return;
  }}
  renders.forEach((item) => {{
    const option = document.createElement('option');
    option.value = item.path;
    option.dataset.url = item.url;
    option.textContent = item.label;
    select.appendChild(option);
  }});
  let chosen = gsRenderUserSelected ? renders.find((item) => item.path === previous) : null;
  if (!chosen) chosen = renders.find((item) => item.path === previous) || renders[renders.length - 1];
  if (!gsRenderUserSelected && chosen.path !== previous) gsRenderUserSelected = false;
  select.value = chosen.path;
  selectedGsRenderPath = chosen.path;
  select.disabled = false;
  setSceneTrajectoryVideo(chosen.url, chosen.path);
}}

function selectedVideoUrl() {{
  const select = $('videoSelect');
  if (!select || select.disabled) return '';
  const option = select.options[select.selectedIndex];
  return option ? option.dataset.url || '' : '';
}}

function setVideoSource(url) {{
  if (!url) return;
  if ($('videoPreview').src.indexOf(url) < 0) {{
    $('videoPreview').src = url;
    $('videoPreview').load();
  }}
}}

function onVideoSelect() {{
  const select = $('videoSelect');
  selectedVideoPath = select.value;
  setVideoSource(selectedVideoUrl());
}}

async function onPdSelect() {{
  const select = $('pdSelect');
  if (!select || select.disabled) return;
  selectedPdPath = select.value;
  pdUserSelected = true;
  await loadPlyInto('step1', selectedPdPath);
}}

async function onGsSelect() {{
  const select = $('gsSelect');
  if (!select || select.disabled) return;
  selectedGsPath = select.value;
  gsUserSelected = true;
  await loadPlyInto('gs', selectedGsPath);
  setSceneViewMode('gs');
}}

function onGsRenderSelect() {{
  const select = $('gsRenderSelect');
  if (!select || select.disabled) return;
  selectedGsRenderPath = select.value;
  gsRenderUserSelected = true;
  const option = select.options[select.selectedIndex];
  setSceneTrajectoryVideo(option ? option.dataset.url || '' : '', selectedGsRenderPath);
  setSceneViewMode('video');
}}

function resetSequence(announce=false) {{
  currentGeneratedJobId = null;
  currentReconJobId = null;
  currentScene = null;
  selectedVideoPath = '';
  selectedPdPath = '';
  selectedGsPath = '';
  selectedGsRenderPath = '';
  pdUserSelected = false;
  gsUserSelected = false;
  gsRenderUserSelected = false;
  renderVideoSelect(null);
  renderPdSelect(null);
  renderGsSelect(null);
  renderGsRenderSelect(null);
  $('videoPreview').removeAttribute('src');
  $('videoPreview').load();
  sceneTrajectoryUrl = '';
  const sceneVideo = $('sceneVideoPreview');
  if (sceneVideo) {{
    sceneVideo.removeAttribute('src');
    sceneVideo.load();
  }}
  clearSceneClouds();
  applySceneDisplay();
  updateSceneMode();
  if (announce) setStatus('new scene');
}}

function onSampleChange() {{
  const item = samples.find((sample) => sample.path === $('sampleSelect').value);
  if (!item) return;
  resetSequence();
  $('inputPreview').src = item.url;
  if (!$('imageInput').files.length && item.prompt) {{
    $('promptInput').value = item.prompt;
  }}
}}

function collectForm() {{
  const fd = new FormData();
  const image = $('imageInput').files[0];
  if (image) fd.append('image', image);
  fd.append('sample_image', $('sampleSelect').value);
  fd.append('ar_continue', $('arContinueInput').checked ? '1' : '0');
  if ($('arContinueInput').checked && currentGeneratedJobId) fd.append('continue_from_job', currentGeneratedJobId);
  fd.append('prompt', $('promptInput').value);
  fd.append('duration', $('durationInput').value);
  fd.append('fps', $('fpsInput').value);
  fd.append('trajectory_mode', $('trajectoryMode').value);
  fd.append('preset', $('presetSelect').value);
  fd.append('operations', operations.join(' '));
  fd.append('strength', $('strengthInput').value);
  fd.append('use_dmd', $('dmdInput').checked ? '1' : '0');
  fd.append('export_sparse_cache_ply', $('step1CloudInput').checked ? '1' : '0');
  fd.append('sparse_cache_max_points', $('sparsePointsInput').value);
  fd.append('use_moge_scale', $('mogeInput').checked ? '1' : '0');
  fd.append('experiment', $('experimentInput').value);
  fd.append('checkpoint_dir', $('checkpointInput').value);
  fd.append('resolution', $('resolutionInput').value);
  fd.append('pose_scale', $('poseScaleInput').value);
  fd.append('seed', $('seedInput').value);
  fd.append('sampling_steps', $('stepsInput').value);
  return fd;
}}

function collectWorkerForm() {{
  const fd = new FormData();
  fd.append('use_dmd', $('dmdInput').checked ? '1' : '0');
  fd.append('export_sparse_cache_ply', $('step1CloudInput').checked ? '1' : '0');
  fd.append('sparse_cache_max_points', $('sparsePointsInput').value);
  fd.append('use_moge_scale', $('mogeInput').checked ? '1' : '0');
  fd.append('experiment', $('experimentInput').value);
  fd.append('checkpoint_dir', $('checkpointInput').value);
  return fd;
}}

function workerIsLoading() {{
  return lastWorkerStatus && lastWorkerStatus.running && ['starting', 'loading', 'offloading', 'moving_gpu'].includes(lastWorkerStatus.status);
}}

function renderWorkerStatus(data) {{
  lastWorkerStatus = data || {{ status: 'stopped', running: false }};
  const state = lastWorkerStatus.status || 'stopped';
  const message = lastWorkerStatus.message || lastWorkerStatus.error || '';
  const el = $('workerStatus');
  el.textContent = message ? `${{state}}: ${{message}}` : state;
  el.className = `worker-state ${{state}}`;
  const btn = $('preloadBtn');
  const releaseBtn = $('releaseBtn');
  const workerBusy = state === 'busy';
  const workerMoving = ['starting', 'loading', 'offloading', 'moving_gpu'].includes(state);
  btn.disabled = Boolean(lastWorkerStatus.running && (workerMoving || workerBusy));
  btn.textContent = lastWorkerStatus.running ? (state === 'ready' ? 'Reload Models' : (state === 'offloaded' ? 'Move Models to GPU' : 'Loading Models')) : 'Preload Models';
  releaseBtn.disabled = Boolean(!lastWorkerStatus.running || workerBusy);
  if (activeLogSource === 'worker') {{
    if (state === 'ready') setStatus('models ready');
    else if (state === 'starting' || state === 'loading') setStatus('preloading models');
    else if (state === 'offloading') setStatus('offloading models');
    else if (state === 'offloaded') setStatus('models on CPU');
    else if (state === 'moving_gpu') setStatus('moving models to GPU');
    else if (state === 'stopped' && message) setStatus(message);
    else if (state === 'failed' || state === 'exited') setStatus('preload failed', true);
  }}
}}

async function refreshWorkerStatus(showLog=false) {{
  const res = await fetch('/api/worker_status');
  const data = await res.json();
  if (!data.ok) return data;
  renderWorkerStatus(data);
  if (showLog && activeLogSource === 'worker') {{
    $('logBox').textContent = data.log || '';
    $('logBox').scrollTop = $('logBox').scrollHeight;
  }}
  return data;
}}

function startWorkerPolling(showLog=false) {{
  if (workerPollTimer) clearInterval(workerPollTimer);
  const tick = () => refreshWorkerStatus(showLog).catch(() => {{}});
  tick();
  workerPollTimer = setInterval(tick, 3000);
}}

async function preloadModels() {{
  if (lastWorkerStatus && lastWorkerStatus.running && lastWorkerStatus.status === 'offloaded') {{
    await moveModelsToGpu('manual');
    return;
  }}
  activeLogSource = 'worker';
  setStatus('preloading models');
  $('logBox').textContent = '';
  $('commandBox').textContent = 'preloaded-worker';
  $('preloadBtn').disabled = true;
  try {{
    const res = await fetch('/api/preload_worker', {{ method: 'POST', body: collectWorkerForm() }});
    const data = await res.json();
    if (!data.ok) {{
      setStatus('preload failed', true);
      $('logBox').textContent = data.error || 'failed';
      showToast('Failed to preload models: ' + (data.error || 'unknown error'), 'error');
      renderWorkerStatus(lastWorkerStatus);
      return;
    }}
    renderWorkerStatus(data);
    startWorkerPolling(true);
    showToast('Models preloading...', 'info', 2000);
  }} catch (err) {{
    setStatus('preload failed', true);
    $('logBox').textContent = err && err.message ? err.message : String(err);
    showToast('Failed to preload models: ' + (err.message || 'network error'), 'error');
    renderWorkerStatus(lastWorkerStatus);
  }}
}}

function formatWorkerTiming(label, data) {{
  const result = data && data.command_result;
  if (!result) return '';
  const elapsed = Number(result.elapsed_sec || 0).toFixed(2);
  const allocated = result.cuda_allocated_mb !== undefined ? `, allocated ${{result.cuda_allocated_mb}} MB` : '';
  const reserved = result.cuda_reserved_mb !== undefined ? `, reserved ${{result.cuda_reserved_mb}} MB` : '';
  return `${{label}}: ${{elapsed}}s${{allocated}}${{reserved}}`;
}}

function workerCommandForm() {{
  const fd = new FormData();
  fd.append('x', '1');
  return fd;
}}

async function offloadModelsForRecon() {{
  activeLogSource = 'worker';
  setStatus('offloading models');
  $('commandBox').textContent = 'offload-preloaded-worker';
  $('preloadBtn').disabled = true;
  $('releaseBtn').disabled = true;
  const fd = workerCommandForm();
  const res = await fetch('/api/offload_worker', {{ method: 'POST', body: fd }});
  const data = await res.json();
  if (!data.ok) {{
    setStatus('offload failed', true);
    $('logBox').textContent = data.error || 'failed';
    renderWorkerStatus(lastWorkerStatus);
    return null;
  }}
  renderWorkerStatus(data);
  const line = formatWorkerTiming('Offload to CPU', data);
  if (line) $('logBox').textContent = `${{line}}\n${{$('logBox').textContent || ''}}`;
  return data.command_result || {{}};
}}

async function moveModelsToGpu(reason='recon') {{
  activeLogSource = 'worker';
  setStatus('moving models to GPU');
  $('commandBox').textContent = 'move-preloaded-worker-to-gpu';
  $('preloadBtn').disabled = true;
  $('releaseBtn').disabled = true;
  const fd = workerCommandForm();
  const res = await fetch('/api/move_worker_gpu', {{ method: 'POST', body: fd }});
  const data = await res.json();
  if (!data.ok) {{
    setStatus('move to GPU failed', true);
    $('logBox').textContent = data.error || 'failed';
    renderWorkerStatus(lastWorkerStatus);
    return null;
  }}
  renderWorkerStatus(data);
  const line = formatWorkerTiming('Move back to GPU', data);
  if (line) $('logBox').textContent = `${{line}}\n${{$('logBox').textContent || ''}}`;
  if (reason === 'recon') setStatus('models restored');
  return data.command_result || {{}};
}}

async function releaseModels() {{
  activeLogSource = 'worker';
  setStatus('releasing models');
  $('logBox').textContent = '';
  $('commandBox').textContent = 'release-preloaded-worker';
  $('preloadBtn').disabled = true;
  $('releaseBtn').disabled = true;
  const fd = workerCommandForm();
  const res = await fetch('/api/release_worker', {{ method: 'POST', body: fd }});
  const data = await res.json();
  if (!data.ok) {{
    setStatus('release failed', true);
    $('logBox').textContent = data.error || 'failed';
    renderWorkerStatus(lastWorkerStatus);
    return;
  }}
  renderWorkerStatus(data);
  $('logBox').textContent = data.log || '';
  if (data.message) setStatus(data.message);
  startWorkerPolling(false);
}}

async function startGeneration() {{
  await refreshWorkerStatus(false).catch(() => lastWorkerStatus);
  if (lastWorkerStatus && lastWorkerStatus.running && lastWorkerStatus.status === 'offloaded') {{
    const moved = await moveModelsToGpu('generation');
    if (!moved) return;
  }}
  if (workerIsLoading()) {{
    setStatus('models loading');
    showToast('Please wait for models to finish loading', 'info');
    return;
  }}
  const willContinue = nextGenerationContinues();
  selectedPdPath = '';
  selectedGsPath = '';
  selectedGsRenderPath = '';
  pdUserSelected = false;
  gsUserSelected = false;
  gsRenderUserSelected = false;
  activeLogSource = 'job';
  setBusy(true);
  setStatus(willContinue ? 'queued: continuation' : 'queued: new scene');
  $('logBox').textContent = '';
  $('commandBox').textContent = '';
  $('paths').innerHTML = '';
  if (!willContinue) {{
    currentScene = null;
    selectedVideoPath = '';
    selectedPdPath = '';
    selectedGsPath = '';
    selectedGsRenderPath = '';
    renderVideoSelect(null);
    renderPdSelect(null);
    renderGsSelect(null);
    renderGsRenderSelect(null);
    $('videoPreview').removeAttribute('src');
    $('videoPreview').load();
  }}
  clearSceneClouds();
  try {{
    const res = await fetch('/api/start_generation', {{ method: 'POST', body: collectForm() }});
    const data = await res.json();
    if (!data.ok) {{
      setStatus('failed', true);
      $('logBox').textContent = data.error || 'failed';
      showToast('Generation failed: ' + (data.error || 'unknown error'), 'error');
      setBusy(false);
      return;
    }}
    currentJobId = data.job.id;
    currentGeneratedJobId = data.job.id;
    updateSceneMode();
    showToast('Generation started', 'success', 2000);
    pollJob(data.job.id, 'generation');
    saveSettings();
  }} catch (err) {{
    setStatus('failed', true);
    $('logBox').textContent = err && err.message ? err.message : String(err);
    showToast('Generation failed: ' + (err.message || 'network error'), 'error');
    setBusy(false);
  }}
}}

async function startReconstruction() {{
  if (!currentGeneratedJobId && !selectedVideoPath) {{
    setStatus('no generated video', true);
    $('logBox').textContent = 'Generate a video or select one from Generated video before reconstruction.';
    showToast('Please generate a video first', 'error');
    return;
  }}
  reconRestoreAfterJob = false;
  activeLogSource = 'job';
  setBusy(true);
  $('reconBtn').disabled = true;
  setStatus('preparing reconstruction');
  $('logBox').textContent = '';
  $('commandBox').textContent = '';
  try {{
    await refreshWorkerStatus(false).catch(() => lastWorkerStatus);
    const workerState = lastWorkerStatus ? lastWorkerStatus.status : '';
    const shouldOffload = Boolean($('offloadReconInput').checked && lastWorkerStatus && lastWorkerStatus.running && ['ready', 'offloaded'].includes(workerState));
    if ($('offloadReconInput').checked && lastWorkerStatus && lastWorkerStatus.running && !['ready', 'offloaded'].includes(workerState)) {{
      setStatus(`worker ${{workerState || 'not ready'}}`, true);
      $('logBox').textContent = `Worker must be ready/offloaded before reconstruction offload. Current state: ${{workerState || 'unknown'}}`;
      showToast('Worker not ready for offload', 'error');
      setBusy(false);
      return;
    }}
    if (shouldOffload) {{
      const offloaded = await offloadModelsForRecon();
      if (!offloaded) {{
        setBusy(false);
        return;
      }}
      reconRestoreAfterJob = true;
    }}
    activeLogSource = 'job';
    setStatus('queued: reconstruction');
    const fd = new FormData();
    fd.append('job_id', currentGeneratedJobId || '');
    if (selectedVideoPath) fd.append('video_path', selectedVideoPath);
    fd.append('force', $('forceReconInput').checked ? '1' : '0');
    const res = await fetch('/api/start_reconstruction', {{ method: 'POST', body: fd }});
    const data = await res.json();
    if (!data.ok) {{
      setStatus('recon failed', true);
      $('logBox').textContent = data.error || 'failed';
      showToast('Reconstruction failed: ' + (data.error || 'unknown error'), 'error');
      if (reconRestoreAfterJob) {{
        reconRestoreAfterJob = false;
        await moveModelsToGpu('recon');
      }}
      setBusy(false);
      return;
    }}
    currentJobId = data.job.id;
    currentReconJobId = data.job.id;
    showToast('Reconstruction started', 'success', 2000);
    pollJob(data.job.id, 'reconstruction');
  }} catch (err) {{
    setStatus('recon failed', true);
    $('logBox').textContent = err && err.message ? err.message : String(err);
    showToast('Reconstruction failed: ' + (err.message || 'network error'), 'error');
    if (reconRestoreAfterJob) {{
      reconRestoreAfterJob = false;
      await moveModelsToGpu('recon');
    }}
    setBusy(false);
  }}
}}

function renderPaths(data) {{
  const paths = $('paths');
  const labels = {{
    input_image: 'Input image',
    trajectory: 'Trajectory npz',
    captions: 'Captions json',
    autoregressive_video: 'AR sequence video',
    expected_autoregressive_video: 'Expected AR sequence video',
    continuation_base_video: 'Continuation base video',
    generated_video: 'Generated segment video',
    expected_video: 'Expected segment video',
    step1_sparse_cache_ply: 'Point cloud sparse cache PLY',
    expected_step1_sparse_cache_ply: 'Expected Point cloud sparse cache PLY',
    reconstruction_video: 'Reconstruction video',
    expected_reconstruction_video: 'Expected reconstruction video',
    ply: 'Gaussian PLY',
    expected_ply: 'Expected PLY',
    source_video: 'Source video'
  }};
  paths.innerHTML = '';
  Object.entries(data.outputs || {{}}).forEach(([key, value]) => {{
    const url = data.urls && data.urls[key];
    const a = document.createElement('a');
    a.textContent = `${{labels[key] || key}}: ${{value}}`;
    if (url) a.href = url;
    paths.appendChild(a);
  }});
}}

function disposeSceneCloud(cloud) {{
  if (!sceneWebgl || !cloud) return;
  const gl = sceneWebgl.gl;
  if (cloud.positionBuffer) gl.deleteBuffer(cloud.positionBuffer);
  if (cloud.colorBuffer) gl.deleteBuffer(cloud.colorBuffer);
  if (cloud.sizeBuffer) gl.deleteBuffer(cloud.sizeBuffer);
  if (cloud.alphaBuffer) gl.deleteBuffer(cloud.alphaBuffer);
  if (cloud.axis0Buffer) gl.deleteBuffer(cloud.axis0Buffer);
  if (cloud.axis1Buffer) gl.deleteBuffer(cloud.axis1Buffer);
  if (cloud.axis2Buffer) gl.deleteBuffer(cloud.axis2Buffer);
}}

function clearSceneClouds() {{
  disposeSceneCloud(pointClouds.step1);
  disposeSceneCloud(pointClouds.gs);
  pointClouds = {{ step1: null, gs: null }};
  resetSceneOrbitState();
  const raster = $('gsRasterPreview');
  if (raster) {{
    raster.removeAttribute('src');
    raster.dataset.path = '';
  }}
  setSceneInfo('');
}}

function makePointCloud(kind, plyPath, payload) {{
  const pts = payload.points || [];
  const cols = payload.colors || [];
  const radii = payload.radii || [];
  const opacities = payload.opacities || [];
  const splatScales = payload.scales || [];
  const splatRotations = payload.rotations || [];
  const renderKind = payload.kind || (kind === 'gs' ? 'gaussian' : 'point_cloud');
  const fallback = kind === 'gs' ? [118, 190, 255] : [220, 235, 228];
  const positions = new Float32Array(pts.length * 3);
  const colors = new Float32Array(pts.length * 3);
  const sizes = new Float32Array(pts.length);
  const alphas = new Float32Array(pts.length);
  const axes0 = new Float32Array(pts.length * 3);
  const axes1 = new Float32Array(pts.length * 3);
  const axes2 = new Float32Array(pts.length * 3);
  for (let idx = 0; idx < pts.length; idx += 1) {{
    const p = pts[idx] || [0, 0, 0];
    const c = cols[idx] || fallback;
    const base = idx * 3;
    positions[base] = Number(p[0]) || 0;
    positions[base + 1] = -(Number(p[1]) || 0);  // Flip Y axis to fix upside-down display
    positions[base + 2] = Number(p[2]) || 0;
    const r = Number(c[0]);
    const g = Number(c[1]);
    const b = Number(c[2]);
    colors[base] = clamp(Number.isFinite(r) ? r : fallback[0], 0, 255) / 255;
    colors[base + 1] = clamp(Number.isFinite(g) ? g : fallback[1], 0, 255) / 255;
    colors[base + 2] = clamp(Number.isFinite(b) ? b : fallback[2], 0, 255) / 255;
    const radius = Number(radii[idx]);
    const alpha = Number(opacities[idx]);
    const safeRadius = Math.max(Number.isFinite(radius) ? radius : 0.0035, 0.0002);
    sizes[idx] = renderKind === 'gaussian' ? safeRadius : 0;
    alphas[idx] = renderKind === 'gaussian' ? clamp(Number.isFinite(alpha) ? alpha : 0.65, 0.02, 1.0) : 1.0;
    if (renderKind === 'gaussian') {{
      const scale = splatScales[idx] || [safeRadius, safeRadius, safeRadius];
      const q = splatRotations[idx] || [1, 0, 0, 0];
      let qw = Number(q[0]);
      let qx = Number(q[1]);
      let qy = Number(q[2]);
      let qz = Number(q[3]);
      const qLen = Math.hypot(qw, qx, qy, qz);
      if (!Number.isFinite(qLen) || qLen <= 1e-8) {{
        qw = 1; qx = 0; qy = 0; qz = 0;
      }} else {{
        qw /= qLen; qx /= qLen; qy /= qLen; qz /= qLen;
      }}
      const sx = Math.max(Number(scale[0]) || safeRadius, 0.0001);
      const sy = Math.max(Number(scale[1]) || safeRadius, 0.0001);
      const sz = Math.max(Number(scale[2]) || safeRadius, 0.0001);
      const m00 = 1 - 2 * (qy * qy + qz * qz);
      const m01 = 2 * (qx * qy - qw * qz);
      const m02 = 2 * (qx * qz + qw * qy);
      const m10 = 2 * (qx * qy + qw * qz);
      const m11 = 1 - 2 * (qx * qx + qz * qz);
      const m12 = 2 * (qy * qz - qw * qx);
      const m20 = 2 * (qx * qz - qw * qy);
      const m21 = 2 * (qy * qz + qw * qx);
      const m22 = 1 - 2 * (qx * qx + qy * qy);
      axes0[base] = m00 * sx; axes0[base + 1] = -m10 * sx; axes0[base + 2] = m20 * sx;
      axes1[base] = m01 * sy; axes1[base + 1] = -m11 * sy; axes1[base + 2] = m21 * sy;
      axes2[base] = m02 * sz; axes2[base + 1] = -m12 * sz; axes2[base + 2] = m22 * sz;
    }}
  }}
  return {{
    kind,
    renderKind,
    sourcePath: plyPath,
    positions,
    colors,
    sizes,
    alphas,
    axes0,
    axes1,
    axes2,
    count: pts.length,
    total: payload.count || pts.length,
    sampled: payload.sampled || pts.length,
    viewHint: payload.view_hint || null,
    rasterUrl: renderKind === 'gaussian' ? '/api/gs_raster_preview?path=' + encodeURIComponent(plyPath) + '&max_points=1000000' : '',
    positionBuffer: null,
    colorBuffer: null,
    sizeBuffer: null,
    alphaBuffer: null,
    axis0Buffer: null,
    axis1Buffer: null,
    axis2Buffer: null
  }};
}}

async function loadPlyInto(kind, plyPath) {{
  if (!plyPath) return;
  const current = pointClouds[kind];
  if (current && current.sourcePath === plyPath && current.count) return;
  const maxPoints = kind === 'gs' ? 260000 : 360000;
  const path = encodeURIComponent(plyPath);
  setSceneMessage('Loading ' + (kind === 'gs' ? 'GS splats' : 'Point cloud') + '...');
  const res = await fetch('/api/ply_preview?path=' + path + '&max_points=' + maxPoints);
  const payload = await res.json();
  if (payload.ok) {{
    disposeSceneCloud(pointClouds[kind]);
    pointClouds[kind] = makePointCloud(kind, plyPath, payload);
    if (kind === 'gs') {{
      sceneUserControlled = false;
      resetSceneOrbitState();
      initSceneOrbitFromSource(pointClouds[kind], true);
    }}
    applySceneDisplay();
  }} else {{
    setSceneMessage(payload.error || 'Failed to load PLY preview');
  }}
}}

function sceneLabelForVideo(scene, videoPath) {{
  const videos = ((scene && scene.videos) || []).filter((item) => item && item.path);
  const found = videos.find((item) => item.path === videoPath);
  if (found && found.label) return found.label;
  if (videoPath && videoPath.indexOf('ar_sequence') >= 0) return 'Full scene';
  return 'Selected video';
}}

function upsertByPath(list, item) {{
  if (!item || !item.path || !item.url) return list || [];
  const out = Array.isArray(list) ? list.slice() : [];
  const idx = out.findIndex((entry) => entry && entry.path === item.path);
  if (idx >= 0) out[idx] = Object.assign({{}}, out[idx], item);
  else out.push(item);
  return out;
}}

function mergeScenePayload(scene) {{
  if (!scene) return currentScene;
  const merged = Object.assign({{}}, currentScene || {{}}, scene);
  ['videos', 'pd_clouds', 'gs_clouds', 'gs_renders'].forEach((key) => {{
    const incoming = Array.isArray(scene[key]) ? scene[key] : [];
    const existing = currentScene && Array.isArray(currentScene[key]) ? currentScene[key] : [];
    merged[key] = incoming.length ? incoming : existing;
  }});
  currentScene = merged;
  return currentScene;
}}

function mergeGsOutputsIntoScene(data) {{
  const outputs = data.outputs || {{}};
  const urls = data.urls || {{}};
  const gsPath = outputs.ply || outputs.expected_ply || '';
  const gsUrl = urls.ply || urls.expected_ply || '';
  const renderPath = outputs.reconstruction_video || outputs.expected_reconstruction_video || '';
  const renderUrl = urls.reconstruction_video || urls.expected_reconstruction_video || '';
  if (!gsUrl && !renderUrl) return currentScene;
  const scene = currentScene || {{ videos: [], pd_clouds: [], gs_clouds: [], gs_renders: [] }};
  const sourceVideo = outputs.source_video || selectedVideoPath || '';
  const label = sceneLabelForVideo(scene, sourceVideo);
  if (gsUrl) {{
    scene.gs_clouds = upsertByPath(scene.gs_clouds, {{
      label: label + ' GS',
      kind: 'gs',
      path: gsPath,
      exists: true,
      url: gsUrl
    }});
  }}
  if (renderUrl) {{
    scene.gs_renders = upsertByPath(scene.gs_renders, {{
      label: label + ' GS_Render',
      kind: 'gs_render',
      path: renderPath,
      exists: true,
      url: renderUrl
    }});
  }}
  currentScene = scene;
  return scene;
}}

async function maybeLoadPly(data) {{
  const outputs = data.outputs || {{}};
  const urls = data.urls || {{}};
  const step1Path = outputs.step1_sparse_cache_ply || outputs.expected_step1_sparse_cache_ply;
  const gsPath = outputs.ply || outputs.expected_ply;
  const incomingScene = data.scene || null;
  if (incomingScene && (incomingScene.videos || incomingScene.pd_clouds || incomingScene.gs_clouds || incomingScene.gs_renders)) {{
    mergeScenePayload(incomingScene);
  }}
  mergeGsOutputsIntoScene(data);
  await renderPdSelect(currentScene || null, Boolean(selectedPdPath || cloudCount(pointClouds.step1)));
  await renderGsSelect(currentScene || null, Boolean(selectedGsPath || cloudCount(pointClouds.gs)));
  renderGsRenderSelect(currentScene || null, Boolean(selectedGsRenderPath || sceneTrajectoryUrl));
  if (!selectedPdPath && step1Path && (urls.step1_sparse_cache_ply || urls.expected_step1_sparse_cache_ply)) {{
    selectedPdPath = step1Path;
    await loadPlyInto('step1', step1Path);
  }}
  if (gsPath && (urls.ply || urls.expected_ply) && !selectedGsPath) {{
    selectedGsPath = gsPath;
    await loadPlyInto('gs', gsPath);
  }}
}}

async function pollJob(jobId, kind) {{
  if (pollTimer) clearInterval(pollTimer);
  const tick = async () => {{
    const res = await fetch('/api/job?id=' + encodeURIComponent(jobId));
    const data = await res.json();
    if (!data.ok) {{
      setStatus('job missing', true);
      setBusy(false);
      clearInterval(pollTimer);
      showToast('Job not found', 'error');
      return;
    }}
    setStatus(`${{kind}}: ${{data.status}}`, data.status === 'failed');
    $('logBox').textContent = data.log || '';
    $('logBox').scrollTop = $('logBox').scrollHeight;
    $('commandBox').textContent = (data.command || []).join(' ');
    renderPaths(data);
    const sceneUrl = data.kind === 'generation' ? renderVideoSelect(data.scene || null) : '';
    const reconstructionUrl = data.urls && (data.urls.reconstruction_video || data.urls.expected_reconstruction_video || '');
    const videoUrl = sceneUrl || reconstructionUrl || (data.urls && (data.urls.autoregressive_video || data.urls.generated_video));
    setVideoSource(videoUrl);
    await maybeLoadPly(data);
    if (data.status === 'complete' || data.status === 'failed' || data.status === 'cancelled') {{
      clearInterval(pollTimer);
      if (data.status === 'complete') {{
        showToast(`${{kind.charAt(0).toUpperCase() + kind.slice(1)}} completed successfully`, 'success', 3000);
      }} else if (data.status === 'failed') {{
        showToast(`${{kind.charAt(0).toUpperCase() + kind.slice(1)}} failed`, 'error', 4000);
      }} else if (data.status === 'cancelled') {{
        showToast(`${{kind.charAt(0).toUpperCase() + kind.slice(1)}} cancelled`, 'info', 2000);
      }}
      if (kind === 'reconstruction' && reconRestoreAfterJob) {{
        reconRestoreAfterJob = false;
        await moveModelsToGpu('recon');
      }}
      setBusy(false);
      if (kind === 'generation' && data.status === 'complete' && $('autoReconInput').checked) {{
        startReconstruction();
      }}
    }}
  }};
  await tick();
  pollTimer = setInterval(tick, 2500);
}}

async function cancelCurrentJob() {{
  if (!currentJobId) return;
  const fd = new FormData();
  fd.append('job_id', currentJobId);
  await fetch('/api/cancel', {{ method: 'POST', body: fd }});
  setStatus('cancelled');
  setBusy(false);
}}

function clamp(value, min, max) {{
  return Math.max(min, Math.min(max, value));
}}

function resetSceneOrbitState() {{
  sceneYaw = 0;
  scenePitch = -0.18;
  sceneZoom = 1.0;
  sceneOrbitTarget = [0, 0, 0];
  sceneOrbitDistance = 4.4;
  sceneOrbitFov = Math.PI / 4;
  sceneOrbitPath = '';
}}

function initSceneOrbitFromSource(cloud, force=false) {{
  const hint = cloud && cloud.viewHint;
  if (!hint || !hint.eye || !hint.target) {{
    if (force) resetSceneOrbitState();
    return false;
  }}
  if (!force && sceneOrbitPath && sceneOrbitPath === cloud.sourcePath) return true;
  const eye = hint.eye.map(Number);
  const target = hint.target.map(Number);
  if (eye.length < 3 || target.length < 3) return false;
  if (!eye.slice(0, 3).every(Number.isFinite) || !target.slice(0, 3).every(Number.isFinite)) return false;
  const offset = [eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]];
  const distance = Math.hypot(offset[0], offset[1], offset[2]);
  if (!Number.isFinite(distance) || distance <= 1e-6) return false;
  sceneOrbitTarget = target.slice(0, 3);
  sceneOrbitDistance = clamp(distance, 0.05, 40.0);
  sceneOrbitFov = clamp(Number(hint.fov_y) || Math.PI / 4, Math.PI / 8, Math.PI / 2.1);
  sceneYaw = Math.atan2(offset[0], offset[2]);
  scenePitch = Math.asin(clamp(offset[1] / distance, -0.98, 0.98));
  sceneZoom = 1.0;
  sceneOrbitPath = cloud.sourcePath || '';
  return true;
}}

function prepareGsFreeCamera(force=false) {{
  if (sceneViewMode !== 'gs') return false;
  return initSceneOrbitFromSource(pointClouds.gs, force);
}}

function sceneOrbitCamera(distance) {{
  const cp = Math.cos(scenePitch);
  const offset = [
    Math.sin(sceneYaw) * cp * distance,
    Math.sin(scenePitch) * distance,
    Math.cos(sceneYaw) * cp * distance
  ];
  const eye = [
    sceneOrbitTarget[0] + offset[0],
    sceneOrbitTarget[1] + offset[1],
    sceneOrbitTarget[2] + offset[2]
  ];
  const forward = vec3Normalize([-offset[0], -offset[1], -offset[2]]);
  let right = vec3Normalize(vec3Cross(forward, [0, 1, 0]));
  if (Math.hypot(right[0], right[1], right[2]) <= 1e-8) right = [1, 0, 0];
  const up = vec3Normalize(vec3Cross(right, forward));
  return {{ offset, eye, forward, right, up }};
}}

function applySceneDisplay() {{
  const canvas = $('pointCanvas');
  const video = $('sceneVideoPreview');
  const raster = $('gsRasterPreview');
  if (!canvas || !video || !raster) return;
  const gs = pointClouds.gs;
  const showVideo = sceneViewMode === 'video' && Boolean(sceneTrajectoryUrl);
  const showRaster = !showVideo && sceneViewMode === 'gs' && gs && gs.rasterUrl && !sceneUserControlled;
  canvas.classList.toggle('hidden', showVideo);
  video.classList.toggle('hidden', !showVideo);
  raster.classList.toggle('hidden', !showRaster);
  if (showRaster && raster.dataset.path !== gs.sourcePath) {{
    raster.dataset.path = gs.sourcePath;
    raster.src = gs.rasterUrl;
  }}
  if (showVideo && video.src.indexOf(sceneTrajectoryUrl) < 0) {{
    video.src = sceneTrajectoryUrl;
    video.load();
  }}
}}

function setSceneTrajectoryVideo(url, path='') {{
  sceneTrajectoryUrl = url || '';
  if (path) selectedGsRenderPath = path;
  applySceneDisplay();
}}

function setSceneViewMode(mode) {{
  sceneViewMode = mode;
  if (mode === 'gs') {{
    sceneUserControlled = false;
    resetSceneOrbitState();
    initSceneOrbitFromSource(pointClouds.gs, true);
  }}
  document.querySelectorAll('[data-scene-view]').forEach((btn) => {{
    btn.classList.toggle('active', btn.dataset.sceneView === mode);
  }});
  applySceneDisplay();
}}

function cloudCount(cloud) {{
  return cloud && cloud.count ? cloud.count : 0;
}}

function formatCount(count) {{
  return Number(count || 0).toLocaleString();
}}

function setSceneMessage(text) {{
  const el = $('sceneEmpty');
  if (!el) return;
  el.textContent = text || '';
  el.classList.toggle('hidden', !text);
}}

function setSceneInfo(text) {{
  const el = $('sceneInfo');
  if (!el) return;
  el.textContent = text || '';
  el.classList.toggle('hidden', !text);
}}

function sceneVisibleSets() {{
  const step1 = pointClouds.step1;
  const gs = pointClouds.gs;
  return {{
    step1: (sceneViewMode === 'gs' || sceneViewMode === 'video') ? null : step1,
    gs: (sceneViewMode === 'step1' || sceneViewMode === 'video') ? null : gs,
    rawStep1: step1,
    rawGs: gs
  }};
}}

function compileSceneShader(gl, type, source) {{
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {{
    throw new Error(gl.getShaderInfoLog(shader) || 'scene shader compile failed');
  }}
  return shader;
}}

function setupSceneWebGL() {{
  if (sceneWebgl) return sceneWebgl;
  const canvas = $('pointCanvas');
  const gl = canvas.getContext('webgl', {{ antialias: true }});
  if (!gl) return null;
  const vertex = `
    attribute vec3 a_position;
    attribute vec3 a_color;
    attribute float a_size;
    attribute float a_alpha;
    attribute vec3 a_axis0;
    attribute vec3 a_axis1;
    attribute vec3 a_axis2;
    uniform mat4 u_matrix;
    uniform vec2 u_viewport;
    uniform float u_pointSize;
    uniform float u_radiusScale;
    uniform float u_maxPointSize;
    uniform float u_gaussianMode;
    varying vec3 v_color;
    varying float v_alpha;
    varying vec3 v_ellipse;
    vec2 clipToPixels(vec4 clip) {{
      float safeW = abs(clip.w) > 0.000001 ? clip.w : 0.000001;
      return (clip.xy / safeW) * 0.5 * u_viewport;
    }}
    void main() {{
      vec4 clip = u_matrix * vec4(a_position, 1.0);
      gl_Position = clip;
      float pointSize = u_pointSize;
      v_ellipse = vec3(1.0, 1.0, 0.0);
      if (u_gaussianMode > 0.5 && a_size > 0.000001) {{
        vec2 centerPx = clipToPixels(clip);
        vec2 d0 = clipToPixels(u_matrix * vec4(a_position + a_axis0, 1.0)) - centerPx;
        vec2 d1 = clipToPixels(u_matrix * vec4(a_position + a_axis1, 1.0)) - centerPx;
        vec2 d2 = clipToPixels(u_matrix * vec4(a_position + a_axis2, 1.0)) - centerPx;
        float covXX = dot(vec3(d0.x, d1.x, d2.x), vec3(d0.x, d1.x, d2.x)) + 0.08;
        float covXY = dot(vec3(d0.x, d1.x, d2.x), vec3(d0.y, d1.y, d2.y));
        float covYY = dot(vec3(d0.y, d1.y, d2.y), vec3(d0.y, d1.y, d2.y)) + 0.08;
        float trace = covXX + covYY;
        float delta = sqrt(max((covXX - covYY) * (covXX - covYY) + 4.0 * covXY * covXY, 0.0));
        float major = sqrt(max((trace + delta) * 0.5, 0.08)) * 2.7;
        float minor = sqrt(max((trace - delta) * 0.5, 0.08)) * 2.7;
        float theta = 0.5 * atan(2.0 * covXY, covXX - covYY);
        pointSize = clamp(major * 2.0, 1.0, u_maxPointSize);
        v_ellipse = vec3(clamp(minor / max(major, 0.0001), 0.08, 1.0), cos(theta), sin(theta));
      }} else {{
        float depth = max(0.18, clip.w);
        float splatSize = a_size * u_radiusScale / depth;
        pointSize = a_size > 0.000001 ? splatSize : u_pointSize;
      }}
      gl_PointSize = clamp(pointSize, 1.0, u_maxPointSize);
      v_color = a_color;
      v_alpha = clamp(a_alpha, 0.0, 1.0);
    }}
  `;
  const fragment = `
    precision mediump float;
    varying vec3 v_color;
    varying float v_alpha;
    varying vec3 v_ellipse;
    void main() {{
      vec2 p = gl_PointCoord * 2.0 - vec2(1.0);
      float x = p.x * v_ellipse.y + p.y * v_ellipse.z;
      float y = -p.x * v_ellipse.z + p.y * v_ellipse.y;
      float ratio = max(v_ellipse.x, 0.08);
      float r2 = x * x + (y * y) / (ratio * ratio);
      if (r2 > 1.0) discard;
      float alpha = v_alpha * exp(-2.25 * r2);
      if (alpha < 0.008) discard;
      gl_FragColor = vec4(v_color, alpha);
    }}
  `;
  const program = gl.createProgram();
  gl.attachShader(program, compileSceneShader(gl, gl.VERTEX_SHADER, vertex));
  gl.attachShader(program, compileSceneShader(gl, gl.FRAGMENT_SHADER, fragment));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {{
    throw new Error(gl.getProgramInfoLog(program) || 'scene program link failed');
  }}
  sceneWebgl = {{
    gl,
    program,
    aPosition: gl.getAttribLocation(program, 'a_position'),
    aColor: gl.getAttribLocation(program, 'a_color'),
    aSize: gl.getAttribLocation(program, 'a_size'),
    aAlpha: gl.getAttribLocation(program, 'a_alpha'),
    aAxis0: gl.getAttribLocation(program, 'a_axis0'),
    aAxis1: gl.getAttribLocation(program, 'a_axis1'),
    aAxis2: gl.getAttribLocation(program, 'a_axis2'),
    uMatrix: gl.getUniformLocation(program, 'u_matrix'),
    uViewport: gl.getUniformLocation(program, 'u_viewport'),
    uPointSize: gl.getUniformLocation(program, 'u_pointSize'),
    uRadiusScale: gl.getUniformLocation(program, 'u_radiusScale'),
    uMaxPointSize: gl.getUniformLocation(program, 'u_maxPointSize'),
    uGaussianMode: gl.getUniformLocation(program, 'u_gaussianMode')
  }};
  return sceneWebgl;
}}

function mat4Perspective(fov, aspect, near, far) {{
  const f = 1 / Math.tan(fov / 2);
  const nf = 1 / (near - far);
  return [
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, (2 * far * near) * nf, 0
  ];
}}

function mat4Multiply(a, b) {{
  const out = new Array(16).fill(0);
  for (let row = 0; row < 4; row += 1) {{
    for (let col = 0; col < 4; col += 1) {{
      for (let k = 0; k < 4; k += 1) out[col * 4 + row] += a[k * 4 + row] * b[col * 4 + k];
    }}
  }}
  return out;
}}

function mat4Translate(z) {{
  return [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,z,1];
}}

function mat4RotX(a) {{
  const c = Math.cos(a), s = Math.sin(a);
  return [1,0,0,0, 0,c,s,0, 0,-s,c,0, 0,0,0,1];
}}

function mat4RotY(a) {{
  const c = Math.cos(a), s = Math.sin(a);
  return [c,0,-s,0, 0,1,0,0, s,0,c,0, 0,0,0,1];
}}

function vec3Normalize(v) {{
  const len = Math.hypot(v[0], v[1], v[2]);
  return len > 1e-8 ? [v[0] / len, v[1] / len, v[2] / len] : [0, 0, 0];
}}

function vec3Cross(a, b) {{
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0]
  ];
}}

function vec3Dot(a, b) {{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}}

function mat4LookAt(eye, target, up) {{
  const z = vec3Normalize([eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]]);
  let x = vec3Normalize(vec3Cross(up, z));
  if (Math.hypot(x[0], x[1], x[2]) <= 1e-8) x = [1, 0, 0];
  const y = vec3Cross(z, x);
  return [
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -vec3Dot(x, eye), -vec3Dot(y, eye), -vec3Dot(z, eye), 1
  ];
}}

function sceneViewForCloud(cloud, gaussian, w, h) {{
  const hint = gaussian && !sceneUserControlled ? cloud.viewHint : null;
  if (hint && hint.eye && hint.target && hint.up) {{
    const fov = clamp(Number(hint.fov_y) || Math.PI / 4, Math.PI / 8, Math.PI / 2.1);
    return {{
      projection: mat4Perspective(fov, w / h, 0.001, 40),
      view: mat4LookAt(hint.eye, hint.target, hint.up),
      cameraMode: 'source'
    }};
  }}
  if (gaussian && cloud.viewHint && initSceneOrbitFromSource(cloud, false)) {{
    const distance = sceneOrbitDistance / clamp(sceneZoom, 0.25, 5.0);
    const camera = sceneOrbitCamera(distance);
    return {{
      projection: mat4Perspective(sceneOrbitFov, w / h, 0.001, 60),
      view: mat4LookAt(camera.eye, sceneOrbitTarget, camera.up),
      cameraMode: 'source-orbit'
    }};
  }}
  const projection = mat4Perspective(Math.PI / 4, w / h, 0.01, 100);
  const distance = 4.4 / clamp(sceneZoom, 0.25, 5.0);
  const view = mat4Multiply(mat4Translate(-distance), mat4Multiply(mat4RotX(scenePitch), mat4RotY(sceneYaw)));
  return {{ projection, view, cameraMode: 'orbit' }};
}}

function resizeSceneCanvas(canvas) {{
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.floor(rect.width * dpr));
  const height = Math.max(1, Math.floor(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {{
    canvas.width = width;
    canvas.height = height;
  }}
  return {{ width, height, dpr }};
}}

function uploadSceneCloud(cloud) {{
  if (!cloud || !sceneWebgl) return;
  const gl = sceneWebgl.gl;
  if (!cloud.positionBuffer) cloud.positionBuffer = gl.createBuffer();
  if (!cloud.colorBuffer) cloud.colorBuffer = gl.createBuffer();
  if (!cloud.sizeBuffer) cloud.sizeBuffer = gl.createBuffer();
  if (!cloud.alphaBuffer) cloud.alphaBuffer = gl.createBuffer();
  if (!cloud.axis0Buffer) cloud.axis0Buffer = gl.createBuffer();
  if (!cloud.axis1Buffer) cloud.axis1Buffer = gl.createBuffer();
  if (!cloud.axis2Buffer) cloud.axis2Buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, cloud.positionBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, cloud.positions, gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, cloud.colorBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, cloud.colors, gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, cloud.sizeBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, cloud.sizes, gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, cloud.alphaBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, cloud.alphas, gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, cloud.axis0Buffer);
  gl.bufferData(gl.ARRAY_BUFFER, cloud.axes0, gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, cloud.axis1Buffer);
  gl.bufferData(gl.ARRAY_BUFFER, cloud.axes1, gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, cloud.axis2Buffer);
  gl.bufferData(gl.ARRAY_BUFFER, cloud.axes2, gl.STATIC_DRAW);
}}

function drawSceneCloud(cloud, viewport, pointSize, state) {{
  if (!cloud || !cloud.count || !sceneWebgl) return;
  if (!cloud.positionBuffer || !cloud.colorBuffer) uploadSceneCloud(cloud);
  const gl = sceneWebgl.gl;
  const x = Math.floor(viewport[0]);
  const yTop = Math.floor(viewport[1]);
  const w = Math.max(1, Math.floor(viewport[2]));
  const h = Math.max(1, Math.floor(viewport[3]));
  const y = state.height - yTop - h;
  gl.viewport(x, y, w, h);
  gl.scissor(x, y, w, h);
  gl.clearColor(0.067, 0.09, 0.098, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.useProgram(sceneWebgl.program);
  const gaussian = cloud.renderKind === 'gaussian';
  if (gaussian) {{
    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.depthMask(false);
  }} else {{
    gl.enable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    gl.depthMask(true);
  }}
  const camera = sceneViewForCloud(cloud, gaussian, w, h);
  const matrix = mat4Multiply(camera.projection, camera.view);
  gl.uniformMatrix4fv(sceneWebgl.uMatrix, false, new Float32Array(matrix));
  gl.uniform2f(sceneWebgl.uViewport, w, h);
  gl.uniform1f(sceneWebgl.uPointSize, pointSize * state.dpr);
  const sourceScale = camera.cameraMode === 'source' || camera.cameraMode === 'source-orbit';
  gl.uniform1f(sceneWebgl.uRadiusScale, gaussian ? Math.min(w, h) * (sourceScale ? 18.0 : 9.0) : 0.0);
  gl.uniform1f(sceneWebgl.uMaxPointSize, gaussian ? 112.0 * state.dpr : 5.0 * state.dpr);
  gl.uniform1f(sceneWebgl.uGaussianMode, gaussian ? 1.0 : 0.0);
  gl.bindBuffer(gl.ARRAY_BUFFER, cloud.positionBuffer);
  gl.enableVertexAttribArray(sceneWebgl.aPosition);
  gl.vertexAttribPointer(sceneWebgl.aPosition, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, cloud.colorBuffer);
  gl.enableVertexAttribArray(sceneWebgl.aColor);
  gl.vertexAttribPointer(sceneWebgl.aColor, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, cloud.sizeBuffer);
  gl.enableVertexAttribArray(sceneWebgl.aSize);
  gl.vertexAttribPointer(sceneWebgl.aSize, 1, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, cloud.alphaBuffer);
  gl.enableVertexAttribArray(sceneWebgl.aAlpha);
  gl.vertexAttribPointer(sceneWebgl.aAlpha, 1, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, cloud.axis0Buffer);
  gl.enableVertexAttribArray(sceneWebgl.aAxis0);
  gl.vertexAttribPointer(sceneWebgl.aAxis0, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, cloud.axis1Buffer);
  gl.enableVertexAttribArray(sceneWebgl.aAxis1);
  gl.vertexAttribPointer(sceneWebgl.aAxis1, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, cloud.axis2Buffer);
  gl.enableVertexAttribArray(sceneWebgl.aAxis2);
  gl.vertexAttribPointer(sceneWebgl.aAxis2, 3, gl.FLOAT, false, 0, 0);
  gl.drawArrays(gl.POINTS, 0, cloud.count);
  gl.depthMask(true);
}}

function drawPointCloud() {{
  const canvas = $('pointCanvas');
  const videoVisible = sceneViewMode === 'video' && Boolean(sceneTrajectoryUrl);
  if (videoVisible) {{
    setSceneMessage('');
    setSceneInfo('GS_Render output');
    requestAnimationFrame(drawPointCloud);
    return;
  }}
  let state;
  let glState;
  try {{
    state = resizeSceneCanvas(canvas);
    glState = setupSceneWebGL();
  }} catch (error) {{
    setSceneMessage(error.message || String(error));
    requestAnimationFrame(drawPointCloud);
    return;
  }}
  if (!glState) {{
    setSceneMessage('浏览器不支持 WebGL');
    requestAnimationFrame(drawPointCloud);
    return;
  }}

  const sets = sceneVisibleSets();
  const step1 = sets.step1;
  const gs = sets.gs;
  const step1Count = cloudCount(step1);
  const gsCount = cloudCount(gs);
  if (!step1Count && !gsCount) {{
    const hasAny = cloudCount(sets.rawStep1) || cloudCount(sets.rawGs);
    let message = hasAny ? 'No ' + (sceneViewMode === 'step1' ? 'Point cloud' : sceneViewMode.toUpperCase()) + ' scene loaded' : 'Point clouds appear after generation / reconstruction';
    if (sceneViewMode === 'gs') message = 'GS scene appears after reconstruction';
    setSceneMessage(message);
    setSceneInfo('');
    const gl = glState.gl;
    gl.viewport(0, 0, state.width, state.height);
    gl.disable(gl.SCISSOR_TEST);
    gl.clearColor(0.067, 0.09, 0.098, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    requestAnimationFrame(drawPointCloud);
    return;
  }}

  setSceneMessage('');
  if (!sceneDragging && !sceneUserControlled && sceneViewMode !== 'gs') sceneYaw += 0.0012;
  const gl = glState.gl;
  gl.enable(gl.DEPTH_TEST);
  gl.enable(gl.SCISSOR_TEST);
  const pad = 10 * state.dpr;
  const full = [pad, pad, state.width - pad * 2, state.height - pad * 2];

  if (gsCount && sceneViewMode === 'gs') {{
    drawSceneCloud(gs, full, 2.6, state);
    const modeText = gs.rasterUrl && !sceneUserControlled ? 'source raster' : (gs.viewHint ? (sceneUserControlled ? 'source free view' : 'source view') : 'orbit view');
    setSceneInfo('GS splats ' + formatCount(gs.sampled || gs.count) + ' / ' + formatCount(gs.total || gs.count) + ' · ' + modeText);
  }} else {{
    drawSceneCloud(step1, full, 2.2, state);
    setSceneInfo('Point cloud ' + formatCount(step1.sampled || step1.count) + ' / ' + formatCount(step1.total || step1.count));
  }}
  requestAnimationFrame(drawPointCloud);
}}


function setupSceneCanvasControls() {{
  const canvas = $('pointCanvas');
  canvas.addEventListener('pointerdown', (event) => {{
    event.preventDefault();
    if (sceneViewMode === 'gs') prepareGsFreeCamera(!sceneUserControlled);
    sceneDragging = true;
    scenePointerMode = (event.button === 1 || event.button === 2 || event.shiftKey) ? 'pan' : 'orbit';
    sceneUserControlled = true;
    applySceneDisplay();
    sceneLastPointer = {{ x: event.clientX, y: event.clientY }};
    canvas.classList.add('dragging');
    canvas.setPointerCapture(event.pointerId);
  }});
  canvas.addEventListener('pointermove', (event) => {{
    if (!sceneDragging) return;
    const dx = event.clientX - sceneLastPointer.x;
    const dy = event.clientY - sceneLastPointer.y;
    sceneLastPointer = {{ x: event.clientX, y: event.clientY }};
    if (scenePointerMode === 'pan' && sceneViewMode === 'gs' && pointClouds.gs && pointClouds.gs.viewHint) {{
      const distance = sceneOrbitDistance / clamp(sceneZoom, 0.25, 5.0);
      const camera = sceneOrbitCamera(distance);
      const scale = Math.max(0.0004, distance * 0.0016);
      sceneOrbitTarget = [
        sceneOrbitTarget[0] - camera.right[0] * dx * scale + camera.up[0] * dy * scale,
        sceneOrbitTarget[1] - camera.right[1] * dx * scale + camera.up[1] * dy * scale,
        sceneOrbitTarget[2] - camera.right[2] * dx * scale + camera.up[2] * dy * scale
      ];
    }} else {{
      sceneYaw += dx * 0.008;
      scenePitch = clamp(scenePitch + dy * 0.006, -1.35, 1.35);
    }}
  }});
  const stopDrag = (event) => {{
    sceneDragging = false;
    scenePointerMode = 'orbit';
    canvas.classList.remove('dragging');
    if (event && canvas.hasPointerCapture && canvas.hasPointerCapture(event.pointerId)) {{
      canvas.releasePointerCapture(event.pointerId);
    }}
  }};
  canvas.addEventListener('pointerup', stopDrag);
  canvas.addEventListener('pointercancel', stopDrag);
  canvas.addEventListener('contextmenu', (event) => event.preventDefault());
  canvas.addEventListener('wheel', (event) => {{
    event.preventDefault();
    if (sceneViewMode === 'gs') prepareGsFreeCamera(!sceneUserControlled);
    sceneUserControlled = true;
    applySceneDisplay();
    sceneZoom = clamp(sceneZoom * Math.exp(-event.deltaY * 0.0012), 0.25, 5.0);
  }}, {{ passive: false }});
}}


$('trajectoryMode').addEventListener('change', updateMode);
$('sampleSelect').addEventListener('change', onSampleChange);
$('imageInput').addEventListener('change', () => {{
  const file = $('imageInput').files[0];
  if (!file) return;
  resetSequence();
  $('inputPreview').src = URL.createObjectURL(file);
}});
document.querySelectorAll('[data-op]').forEach((btn) => btn.addEventListener('click', () => addOp(btn.dataset.op)));
$('clearOps').addEventListener('click', () => {{ operations = []; renderOps(); }});
$('sampleOps').addEventListener('click', () => {{ operations = ['w','w','left','w','d','up','w','right','w']; renderOps(); }});
$('preloadBtn').addEventListener('click', preloadModels);
$('releaseBtn').addEventListener('click', releaseModels);
$('resetSceneBtn').addEventListener('click', () => resetSequence(true));
$('videoSelect').addEventListener('change', onVideoSelect);
$('pdSelect').addEventListener('change', onPdSelect);
$('gsSelect').addEventListener('change', onGsSelect);
$('gsRenderSelect').addEventListener('change', onGsRenderSelect);
document.querySelectorAll('[data-scene-view]').forEach((btn) => btn.addEventListener('click', () => setSceneViewMode(btn.dataset.sceneView)));
setupSceneCanvasControls();
$('arContinueInput').addEventListener('change', updateSceneMode);
$('generateBtn').addEventListener('click', startGeneration);
$('reconBtn').addEventListener('click', startReconstruction);
$('cancelBtn').addEventListener('click', cancelCurrentJob);
$('helpBtn').addEventListener('click', () => {{
  $('helpModal').classList.add('active');
}});
$('closeHelp').addEventListener('click', () => {{
  $('helpModal').classList.remove('active');
}});
$('helpModal').addEventListener('click', (e) => {{
  if (e.target === $('helpModal')) {{
    $('helpModal').classList.remove('active');
  }}
}});
window.addEventListener('keydown', (event) => {{
  const target = event.target;
  if (target && ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)) {{
    // Allow Ctrl+S in input fields
    if (event.key === 's' && (event.ctrlKey || event.metaKey)) {{
      event.preventDefault();
      saveSettings();
      showToast('Settings saved', 'success', 2000);
    }}
    return;
  }}

  // Global shortcuts
  if (event.key === 'g' || event.key === 'G') {{
    event.preventDefault();
    if (!$('generateBtn').disabled) {{
      startGeneration();
    }}
  }} else if (event.key === 'r' || event.key === 'R') {{
    event.preventDefault();
    if (!$('reconBtn').disabled) {{
      startReconstruction();
    }}
  }} else if (event.key === 'Escape') {{
    event.preventDefault();
    if (currentJobId) {{
      cancelCurrentJob();
    }}
  }} else if (event.key === 's' && (event.ctrlKey || event.metaKey)) {{
    event.preventDefault();
    saveSettings();
    showToast('Settings saved', 'success', 2000);
  }}

  // Camera operations in keyboard mode
  const map = {{'w':'w','s':'s','a':'a','d':'d','ArrowUp':'up','ArrowDown':'down','ArrowLeft':'left','ArrowRight':'right'}};
  if ($('trajectoryMode').value === 'keyboard' && map[event.key]) {{
    event.preventDefault();
    addOp(map[event.key]);
  }}
}});

updateMode();
loadSamples();
renderVideoSelect(null);
updateSceneMode();
renderWorkerStatus(lastWorkerStatus);
startWorkerPolling(false);
renderOps();
drawPointCloud();

// Load saved settings
fetch('/api/settings')
  .then(res => res.json())
  .then(data => {{
    if (data.ok && data.settings) {{
      loadSettings(data.settings);
    }}
  }})
  .catch(() => {{}});

// Auto-save settings on change (debounced)
let saveTimer = null;
function autoSave() {{
  if (saveTimer) clearTimeout(saveTimer);
  saveTimer = setTimeout(saveSettings, 1000);
}}
['promptInput', 'durationInput', 'fpsInput', 'strengthInput', 'experimentInput',
 'checkpointInput', 'resolutionInput', 'poseScaleInput', 'seedInput', 'stepsInput',
 'sparsePointsInput'].forEach(id => {{
  const el = $(id);
  if (el) el.addEventListener('input', autoSave);
}});
['trajectoryMode', 'presetSelect', 'dmdInput', 'step1CloudInput', 'autoReconInput',
 'offloadReconInput', 'arContinueInput', 'mogeInput', 'forceReconInput'].forEach(id => {{
  const el = $(id);
  if (el) el.addEventListener('change', autoSave);
}});

// Keyboard shortcuts info
console.log('%cLyra-2 UI Keyboard Shortcuts:', 'font-weight: bold; font-size: 14px; color: #247c62');
console.log('  G - Start generation');
console.log('  R - Start reconstruction');
console.log('  Esc - Cancel current job');
console.log('  W/S/A/D + Arrow keys - Add camera operations (in keyboard mode)');
console.log('  Ctrl+S - Save settings manually');

</script>
</body>
</html>"""


class LyraDemoHandler(BaseHTTPRequestHandler):
    server_version = "Lyra2DemoUI/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[ui] {self.address_string()} - {fmt % args}")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        try:
            if parsed.path == "/":
                _text_response(self, _index_html())
            elif parsed.path == "/gs_viewer.html":
                gs_viewer_path = ROOT / "gs_viewer.html"
                if gs_viewer_path.is_file():
                    _text_response(self, gs_viewer_path.read_text(encoding="utf-8"))
                else:
                    self.send_error(404, "gs_viewer.html not found")
            elif parsed.path == "/api/samples":
                _json_response(self, {"ok": True, "samples": _sample_images()})
            elif parsed.path == "/api/settings":
                _json_response(self, {"ok": True, "settings": _load_ui_settings()})
            elif parsed.path == "/api/worker_status":
                payload = _read_worker_status()
                payload["ok"] = True
                _json_response(self, payload)
            elif parsed.path == "/api/job":
                query = urllib.parse.parse_qs(parsed.query)
                job_id = query.get("id", [""])[0]
                with jobs_lock:
                    job = jobs.get(job_id)
                if job is None:
                    _error_response(self, f"unknown job id: {job_id}", status=404)
                else:
                    _json_response(self, _job_payload(job))
            elif parsed.path == "/api/ply_preview":
                query = urllib.parse.parse_qs(parsed.query)
                path = _safe_relpath(query.get("path", [""])[0])
                try:
                    max_points = int(query.get("max_points", ["360000"])[0] or 360000)
                except ValueError:
                    max_points = 360000
                max_points = max(1000, min(max_points, 360000))
                preview = _ply_preview(path, max_points=max_points)
                preview["ok"] = True
                _json_response(self, preview)
            elif parsed.path == "/api/gs_raster_preview":
                query = urllib.parse.parse_qs(parsed.query)
                path = _safe_relpath(query.get("path", [""])[0])
                try:
                    max_points = int(query.get("max_points", ["700000"])[0] or 700000)
                except ValueError:
                    max_points = 700000
                try:
                    width = int(query.get("width", ["832"])[0] or 832)
                    height = int(query.get("height", ["480"])[0] or 480)
                except ValueError:
                    width, height = 832, 480
                png = _gs_raster_preview(path, max_points=max_points, width=width, height=height)
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(png)))
                self.end_headers()
                self.wfile.write(png)
            elif parsed.path == "/file":
                query = urllib.parse.parse_qs(parsed.query)
                path = _safe_relpath(query.get("path", [""])[0])
                if not path.is_file():
                    self.send_error(404)
                    return
                mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(path.stat().st_size))
                self.end_headers()
                with path.open("rb") as f:
                    shutil.copyfileobj(f, self.wfile)
            else:
                self.send_error(404)
        except Exception as exc:
            _error_response(self, str(exc), status=500)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        try:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers.get("Content-Type", "")},
            )
            if parsed.path == "/api/preload_worker":
                payload = _start_preload_worker(form)
                payload["ok"] = True
                _json_response(self, payload)
            elif parsed.path == "/api/release_worker":
                payload = _release_preload_worker()
                payload["ok"] = True
                _json_response(self, payload)
            elif parsed.path == "/api/offload_worker":
                payload = _offload_worker_to_cpu()
                payload["ok"] = True
                _json_response(self, payload)
            elif parsed.path == "/api/move_worker_gpu":
                payload = _move_worker_to_gpu()
                payload["ok"] = True
                _json_response(self, payload)
            elif parsed.path == "/api/start_generation":
                job = _start_generation(form)
                _json_response(self, {"ok": True, "job": _job_payload(job)})
            elif parsed.path == "/api/start_reconstruction":
                job = _start_reconstruction(form)
                _json_response(self, {"ok": True, "job": _job_payload(job)})
            elif parsed.path == "/api/cancel":
                job_id = _field_to_text(form["job_id"] if "job_id" in form else None, "").strip()
                ok = _cancel_job(job_id)
                _json_response(self, {"ok": ok})
            elif parsed.path == "/api/save_settings":
                settings_json = _field_to_text(form["settings"] if "settings" in form else None, "{}")
                try:
                    settings = json.loads(settings_json)
                    _save_ui_settings(settings)
                    _json_response(self, {"ok": True})
                except Exception as exc:
                    _error_response(self, str(exc), status=400)
            else:
                self.send_error(404)
        except Exception as exc:
            _error_response(self, str(exc), status=500)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the local Lyra-2 demo UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    _adopt_existing_worker()
    server = ThreadingHTTPServer((args.host, args.port), LyraDemoHandler)
    print(f"Lyra-2 demo UI: http://{args.host}:{args.port}")
    print(f"Workspace: {ROOT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
