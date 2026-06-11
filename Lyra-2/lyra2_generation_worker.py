#!/usr/bin/env python3
"""Persistent Lyra-2 generation worker for the local demo UI.

The UI starts this process once to pay the expensive checkpoint-loading cost up
front. Tasks are exchanged through small JSON files in a control directory.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from lyra_2._ext.imaginaire.utils import log, misc
from lyra_2._ext.imaginaire.visualize.video import save_img_or_video
from lyra_2._src.inference.lyra2_ar_inference import run_lyra2_sample, safe_to
from lyra_2._src.inference.lyra2_custom_traj_inference import _apply_dmd_defaults, load_trajectory
from lyra_2._src.inference.lyra2_zoomgs_inference import _da3_infer_depth_intrinsics_single
from lyra_2._src.utils.model_loader import load_model_from_checkpoint

ROOT = Path(__file__).resolve().parent


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _append_log(path: str | Path, line: str) -> None:
    with Path(path).open("a", encoding="utf-8", errors="replace") as f:
        f.write(line)
        if not line.endswith("\n"):
            f.write("\n")


def _base_args(config: dict) -> SimpleNamespace:
    args = SimpleNamespace()
    args.experiment = config.get("experiment", "lyra2")
    args.checkpoint_dir = config.get("checkpoint_dir", "checkpoints/model")
    args.resolution = config.get("resolution", "480,832")
    args.use_dmd = bool(config.get("use_dmd", True))
    args.use_moge_scale = bool(config.get("use_moge_scale", True))
    args.guidance = float(config.get("guidance", 5.0))
    args.shift = float(config.get("shift", 5.0))
    args.num_sampling_step = int(config.get("num_sampling_step", 35))
    args.seed = int(config.get("seed", 1))
    args.fps = int(config.get("fps", 16))
    args.num_frames = int(config.get("num_frames", 81))
    args.pose_scale = float(config.get("pose_scale", 1.1))
    args.context_parallel_size = 1
    args.lora_paths = None
    args.lora_weights = None
    args.offload = bool(config.get("offload", False))
    args.offload_when_prompt = bool(config.get("offload_when_prompt", False))
    args.debug = False
    args.depth_backend = "da3"
    args.da3_model_name = config.get("da3_model_name", "depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    args.da3_model_path_custom = config.get("da3_model_path_custom")
    args.da3_frame_interval = int(config.get("da3_frame_interval", 8))
    args.da3_max_history_frames = int(config.get("da3_max_history_frames", 10))
    args.da3_include_ar_chunk_last_frames = bool(config.get("da3_include_ar_chunk_last_frames", False))
    args.da3_use_predicted_pose = bool(config.get("da3_use_predicted_pose", False))
    args.da3_predicted_pose_continuation = bool(config.get("da3_predicted_pose_continuation", False))
    args.ablate_same_t5 = False
    args.use_dmd_scheduler = False
    args.warp_chunk_size = config.get("warp_chunk_size")
    args.num_retrieval_views = int(config.get("num_retrieval_views", 1))
    args.disable_cache_update = bool(config.get("disable_cache_update", False))
    args.multiview_ids = None
    args.offload_da3_diffusion = bool(config.get("offload_da3_diffusion", False))
    args.export_sparse_cache_ply = bool(config.get("export_sparse_cache_ply", True))
    args.sparse_cache_max_points = int(config.get("sparse_cache_max_points", 200000))
    args.prompt_suffix = config.get("prompt_suffix", "")
    return args


class GenerationWorker:
    def __init__(self, control_dir: Path, config: dict) -> None:
        self.control_dir = control_dir
        self.tasks_dir = control_dir / "tasks"
        self.results_dir = control_dir / "results"
        self.commands_dir = control_dir / "commands"
        self.command_results_dir = control_dir / "command_results"
        self.status_path = control_dir / "status.json"
        self.config = config
        self.args = _base_args(config)
        _apply_dmd_defaults(self.args)
        self.model = None
        self.da3_model = None
        self.moge_model = None
        self.negative_prompt_data = None
        self.desired_dtype = None
        self.desired_device = None
        self.models_on_gpu = False

    def set_status(self, status: str, **extra) -> None:
        payload = {
            "status": status,
            "updated_at": time.time(),
            "pid": os.getpid(),
            "config": self.config,
        }
        payload.update(extra)
        _write_json(self.status_path, payload)

    def load(self) -> None:
        self.set_status("loading", message="loading negative prompt and FramePack checkpoint")
        misc.set_random_seed(seed=int(self.args.seed), by_rank=True)
        self.negative_prompt_data = torch.load(
            "checkpoints/text_encoder/negative_prompt.pt", map_location="cpu", weights_only=False
        )

        experiment_opts = [
            "model.config.use_mp_policy_fsdp=False",
            "model.config.keep_original_net_dtype=False",
        ]
        if self.args.lora_paths:
            experiment_opts += ["model.config.net.postpone_checkpoint=True"]

        self.model, _config = load_model_from_checkpoint(
            config_file="lyra_2/_src/configs/config.py",
            experiment_name=self.args.experiment,
            checkpoint_path=self.args.checkpoint_dir,
            enable_fsdp=False,
            instantiate_ema=False,
            load_ema_to_reg=False,
            experiment_opts=experiment_opts,
        )

        if self.args.lora_paths:
            lora_names = []
            for lora_path in self.args.lora_paths:
                lora_name = self.model.load_lora_weights(lora_path)
                lora_names.append(lora_name)
            self.model.set_weights_and_activate_adapters(lora_names, self.args.lora_weights)
            if hasattr(self.model, "net") and hasattr(self.model.net, "enable_selective_checkpoint"):
                self.model.net.enable_selective_checkpoint(self.model.net.sac_config, self.model.net.blocks)

        self.desired_dtype = self.model.tensor_kwargs.get("dtype", None)
        self.desired_device = self.model.tensor_kwargs.get("device", None)
        if self.desired_dtype is not None:
            self.model.net = self.model.net.to(device=self.desired_device, dtype=self.desired_dtype)
            log.info(f"Casted model.net to dtype={self.desired_dtype}", rank0_only=True)

        assert getattr(self.model.config, "important_start", True) is True
        assert getattr(self.model.config, "encode_video_from_start", True) is True
        assert not getattr(self.model.config, "use_hd_map_cond", False)
        self.model.eval()

        if self.args.warp_chunk_size is not None:
            self.model.config.warp_chunk_size = self.args.warp_chunk_size
            self.model.warp_chunk_size = self.args.warp_chunk_size

        self.set_status("loading", message="loading DA3 model")
        from lyra_2._src.inference.depth_utils import load_da3_model

        da3_device = self.model.tensor_kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.da3_model = load_da3_model(
            da3_model_name=self.args.da3_model_name,
            da3_model_path_custom=self.args.da3_model_path_custom,
            device=da3_device,
        )
        self.da3_model.eval()

        if self.args.use_moge_scale:
            self.set_status("loading", message="loading MoGe model")
            from lyra_2._src.inference.depth_utils import load_moge_model

            moge_device = self.model.tensor_kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            self.moge_model = load_moge_model(moge_device)
            self.moge_model.eval()

        self.models_on_gpu = bool(torch.cuda.is_available() and str(self.desired_device).startswith("cuda"))
        self.set_status("ready", message="models loaded", models_device="gpu" if self.models_on_gpu else "cpu")

    def _cuda_snapshot(self) -> dict:
        if not torch.cuda.is_available():
            return {}
        return {
            "cuda_allocated_mb": round(torch.cuda.memory_allocated() / (1024 ** 2), 1),
            "cuda_reserved_mb": round(torch.cuda.memory_reserved() / (1024 ** 2), 1),
        }

    def _sync_cuda(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def offload_to_cpu(self) -> dict:
        if not self.models_on_gpu:
            return {"action": "offload_cpu", "elapsed_sec": 0.0, "message": "models already on CPU", **self._cuda_snapshot()}
        self.set_status("offloading", message="moving preloaded models to CPU", models_device="moving")
        start = time.time()
        if self.model is not None and getattr(self.model, "net", None) is not None:
            self.model.net = self.model.net.to(device="cpu")
        if self.da3_model is not None:
            self.da3_model.to("cpu")
        if self.moge_model is not None:
            self.moge_model.to("cpu")
        self._sync_cuda()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        elapsed = time.time() - start
        self.models_on_gpu = False
        payload = {"action": "offload_cpu", "elapsed_sec": elapsed, "message": "models offloaded to CPU", **self._cuda_snapshot()}
        status_payload = {k: v for k, v in payload.items() if k != "message"}
        self.set_status("offloaded", message="models on CPU", models_device="cpu", **status_payload)
        return payload

    def move_to_gpu(self) -> dict:
        if self.models_on_gpu:
            return {"action": "move_gpu", "elapsed_sec": 0.0, "message": "models already on GPU", **self._cuda_snapshot()}
        if self.desired_device is None:
            raise RuntimeError("worker has no target GPU device")
        self.set_status("moving_gpu", message="moving preloaded models back to GPU", models_device="moving")
        start = time.time()
        if self.model is not None and getattr(self.model, "net", None) is not None:
            kwargs = {"device": self.desired_device}
            if self.desired_dtype is not None:
                kwargs["dtype"] = self.desired_dtype
            self.model.net = self.model.net.to(**kwargs)
        if self.da3_model is not None:
            self.da3_model.to(self.desired_device)
        if self.moge_model is not None:
            self.moge_model.to(self.desired_device)
        self._sync_cuda()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        elapsed = time.time() - start
        self.models_on_gpu = True
        payload = {"action": "move_gpu", "elapsed_sec": elapsed, "message": "models moved back to GPU", **self._cuda_snapshot()}
        status_payload = {k: v for k, v in payload.items() if k != "message"}
        self.set_status("ready", message="models on GPU", models_device="gpu", **status_payload)
        return payload

    def run_command(self, command: dict) -> dict:
        action = command.get("action")
        if action == "offload_cpu":
            return self.offload_to_cpu()
        if action == "move_gpu":
            return self.move_to_gpu()
        raise ValueError(f"unknown worker command: {action}")

    def run_task(self, task: dict) -> dict:
        if not self.models_on_gpu:
            self.move_to_gpu()
        args = copy.copy(self.args)
        args.input_image_path = task["input_image_path"]
        args.trajectory_path = task["trajectory_path"]
        args.captions_path = task.get("captions_path")
        args.prompt = task.get("prompt", "")
        args.prompt_dir = None
        args.prompt_suffix = task.get("prompt_suffix", "")
        args.output_path = task["output_path"]
        args.num_frames = int(task["num_frames"])
        args.fps = int(task.get("fps", args.fps))
        args.resolution = task.get("resolution", args.resolution)
        args.pose_scale = float(task.get("pose_scale", args.pose_scale))
        args.seed = int(task.get("seed", args.seed))
        args.num_sampling_step = int(task.get("num_sampling_step", args.num_sampling_step))
        args.export_sparse_cache_ply = bool(task.get("export_sparse_cache_ply", True))
        args.sparse_cache_max_points = int(task.get("sparse_cache_max_points", args.sparse_cache_max_points))
        args.sample_start_idx = 0
        args.num_samples = 1

        os.makedirs(args.output_path, exist_ok=True)
        misc.set_random_seed(seed=args.seed, by_rank=True)

        target_h, target_w = [int(x) for x in args.resolution.split(",")]
        N = int(args.num_frames)
        img_path = str(task["input_image_path"])
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        video_path = os.path.join(args.output_path, f"{base_name}.mp4")

        w2cs_T_44, Ks_T_33 = load_trajectory(
            str(task["trajectory_path"]),
            N,
            target_hw=(target_h, target_w),
            pose_scale=args.pose_scale,
        )

        bgr = cv2.imread(img_path)
        if bgr is None:
            raise RuntimeError(f"Cannot read input image: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_t = torch.from_numpy(rgb)

        image_chw01, depth_hw, _K_33_da3, mask_hw = _da3_infer_depth_intrinsics_single(
            da3_model=self.da3_model,
            img_rgb_uint8=rgb_t,
            target_hw=(target_h, target_w),
        )
        H, W = image_chw01.shape[-2:]

        if args.use_moge_scale and self.moge_model is not None:
            from lyra_2._src.inference.depth_utils import moge_infer_depth_intrinsics

            self.moge_model.to(self.desired_device)
            with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
                _, moge_depth_hw, _, moge_mask_hw = moge_infer_depth_intrinsics(
                    self.moge_model,
                    rgb_t,
                    depth_pred_hw=(target_h, target_w),
                    target_hw=(target_h, target_w),
                )

            da3_d = depth_hw.to(moge_depth_hw.device)
            da3_m = mask_hw.to(moge_mask_hw.device)
            valid_mask = (da3_m > 0.5) & (moge_mask_hw > 0.5)
            if valid_mask.sum() > 10:
                inv_da3 = 1.0 / (da3_d[valid_mask] + 1e-6)
                inv_moge = 1.0 / (moge_depth_hw[valid_mask] + 1e-6)
                denominator = (inv_da3 * inv_da3).sum()
                if denominator > 1e-8:
                    scale = (inv_da3 * inv_moge).sum() / denominator
                    if scale > 1e-6:
                        depth_hw = depth_hw / scale.to(depth_hw.device)
            self.moge_model.cpu()
            del moge_depth_hw, moge_mask_hw, da3_d, da3_m
            torch.cuda.empty_cache()
            gc.collect()

        img_bchw = image_chw01.to(device=self.desired_device) * 2.0 - 1.0

        from lyra_2._src.inference.get_t5_emb import get_umt5_embedding, get_umt5_embedding_offloaded

        neg_t5 = misc.to(self.negative_prompt_data["t5_text_embeddings"], **self.model.tensor_kwargs)

        captions_file = args.captions_path if args.captions_path and os.path.isfile(args.captions_path) else None
        use_chunk_captions = False
        single_caption = ""
        if captions_file is not None:
            with open(captions_file, "r", encoding="utf-8") as f:
                captions_dict = json.load(f)
            chunk_keys_int = sorted(int(k) for k in captions_dict)
            chunk_keys_int = [k for k in chunk_keys_int if k < N]
            if len(chunk_keys_int) > 1:
                use_chunk_captions = True
                chunk_keys = torch.tensor(chunk_keys_int, dtype=torch.long, device=self.desired_device)
                chunk_embs = []
                chunk_masks = []
                for ck in chunk_keys_int:
                    cap = captions_dict[str(ck)]
                    if args.prompt_suffix:
                        cap = cap.rstrip() + " " + args.prompt_suffix
                    if args.offload_when_prompt:
                        emb = get_umt5_embedding_offloaded(cap, device=self.desired_device).to(dtype=self.desired_dtype)
                    else:
                        emb = get_umt5_embedding(cap, device=self.desired_device).to(dtype=self.desired_dtype)
                    if emb.dim() == 3:
                        emb = emb[0]
                    S, D = emb.shape
                    S = min(S, 512)
                    D = min(D, 4096)
                    padded_emb = torch.zeros(512, 4096, dtype=self.desired_dtype, device=self.desired_device)
                    padded_emb[:S, :D] = emb[:S, :D]
                    padded_mask = torch.zeros(512, dtype=self.desired_dtype, device=self.desired_device)
                    padded_mask[:S] = 1.0
                    chunk_embs.append(padded_emb)
                    chunk_masks.append(padded_mask)
                t5_chunk_embeddings = torch.stack(chunk_embs).unsqueeze(0)
                t5_chunk_mask = torch.stack(chunk_masks).unsqueeze(0)
                t5_chunk_keys = chunk_keys.unsqueeze(0)
                sample_frame_indices = torch.arange(N, dtype=torch.long, device=self.desired_device).unsqueeze(0)
                t5 = t5_chunk_embeddings[:, 0, :, :]
            else:
                single_caption = captions_dict.get(str(chunk_keys_int[0]), "") if chunk_keys_int else ""

        if not use_chunk_captions:
            caption = args.prompt or single_caption
            if args.prompt_suffix:
                caption = caption.rstrip() + " " + args.prompt_suffix
            if args.offload_when_prompt:
                t5 = get_umt5_embedding_offloaded(caption, device=self.desired_device).to(dtype=self.desired_dtype)
            else:
                t5 = get_umt5_embedding(caption, device=self.desired_device).to(dtype=self.desired_dtype)
            if t5.dim() == 2:
                t5 = t5.unsqueeze(0)
            elif t5.dim() == 3 and t5.shape[0] != 1:
                t5 = t5[:1]

        w2cs_b_t_44 = w2cs_T_44.unsqueeze(0).to(dtype=torch.float32, device=self.desired_device)
        Ks_b_t_33 = Ks_T_33.unsqueeze(0).to(dtype=torch.float32, device=self.desired_device)
        depth_b_thw = depth_hw.unsqueeze(0).unsqueeze(0).repeat(1, N, 1, 1).to(device=self.desired_device)

        data_batch = {
            "video": img_bchw.unsqueeze(2),
            "t5_text_embeddings": t5,
            "neg_t5_text_embeddings": neg_t5,
            "fps": torch.tensor([args.fps], dtype=torch.int32, device=self.desired_device),
            "padding_mask": torch.zeros((1, 1, H, W), dtype=self.model.tensor_kwargs["dtype"], device=self.desired_device),
            "is_preprocessed": torch.tensor([True], dtype=torch.bool, device=self.desired_device),
            "camera_w2c": w2cs_b_t_44,
            "intrinsics": Ks_b_t_33,
            "depth": depth_b_thw,
        }

        if use_chunk_captions:
            data_batch["t5_chunk_keys"] = t5_chunk_keys
            data_batch["t5_chunk_embeddings"] = t5_chunk_embeddings
            data_batch["t5_chunk_mask"] = t5_chunk_mask
            data_batch["sample_frame_indices"] = sample_frame_indices

        skip_keys = {"camera_w2c", "intrinsics", "depth", "t5_chunk_keys", "sample_frame_indices"}
        data_batch = safe_to(
            data_batch,
            device=self.model.tensor_kwargs.get("device", None),
            dtype=self.model.tensor_kwargs.get("dtype", None),
            skip_keys=skip_keys,
        )

        sparse_cache_stem = os.path.join(args.output_path, f"{base_name}_step1") if args.export_sparse_cache_ply else None
        result = run_lyra2_sample(
            self.model,
            data_batch,
            args,
            process_group=None,
            da3_model=self.da3_model,
            show_progress=True,
            log_prefix=f"{base_name}_custom_traj_worker",
            da3_gs_export_stem=sparse_cache_stem,
        )
        if result is None:
            raise RuntimeError("Generation failed")

        video_01 = (result["video"][0].clamp(-1, 1) * 0.5 + 0.5).float().cpu()
        save_img_or_video(video_01, video_path.replace(".mp4", ""), fps=args.fps)
        outputs = {"generated_video": video_path}
        if result.get("sparse_cache_ply"):
            outputs["step1_sparse_cache_ply"] = result["sparse_cache_ply"]

        del result, data_batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return outputs

    def loop(self) -> None:
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.commands_dir.mkdir(parents=True, exist_ok=True)
        self.command_results_dir.mkdir(parents=True, exist_ok=True)
        self.set_status("ready", message="waiting for tasks", models_device="gpu" if self.models_on_gpu else "cpu")
        while True:
            commands = sorted(self.commands_dir.glob("*.json"))
            if commands:
                command_path = commands[0]
                claimed_path = command_path.with_suffix(".running")
                try:
                    command_path.rename(claimed_path)
                except FileNotFoundError:
                    continue
                command = json.loads(claimed_path.read_text(encoding="utf-8"))
                command_id = command.get("command_id", claimed_path.stem)
                result_path = self.command_results_dir / f"{command_id}.json"
                started = time.time()
                try:
                    result = self.run_command(command)
                    result.setdefault("elapsed_sec", time.time() - started)
                    _write_json(result_path, {"ok": True, "command_id": command_id, **result})
                except Exception as exc:
                    _write_json(result_path, {
                        "ok": False,
                        "command_id": command_id,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "elapsed_sec": time.time() - started,
                    })
                    self.set_status("failed", error=str(exc), traceback=traceback.format_exc())
                finally:
                    try:
                        claimed_path.unlink()
                    except FileNotFoundError:
                        pass
                continue

            tasks = sorted(self.tasks_dir.glob("*.json"))
            if not tasks:
                time.sleep(1.0)
                continue
            task_path = tasks[0]
            claimed_path = task_path.with_suffix(".running")
            try:
                task_path.rename(claimed_path)
            except FileNotFoundError:
                continue
            task = json.loads(claimed_path.read_text(encoding="utf-8"))
            job_id = task.get("job_id", claimed_path.stem)
            log_path = Path(task["log_path"])
            result_path = self.results_dir / f"{job_id}.json"
            self.set_status("busy", message=f"running {job_id}", current_job_id=job_id)
            started = time.time()
            try:
                _append_log(log_path, "[worker] using preloaded Lyra-2 models")
                with log_path.open("a", encoding="utf-8", errors="replace") as log_file:
                    with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                        outputs = self.run_task(task)
                _write_json(result_path, {
                    "ok": True,
                    "job_id": job_id,
                    "outputs": outputs,
                    "elapsed_sec": time.time() - started,
                })
            except Exception as exc:
                _append_log(log_path, traceback.format_exc())
                _write_json(result_path, {
                    "ok": False,
                    "job_id": job_id,
                    "error": str(exc),
                    "elapsed_sec": time.time() - started,
                })
            finally:
                try:
                    claimed_path.unlink()
                except FileNotFoundError:
                    pass
                self.set_status("ready", message="waiting for tasks")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent Lyra-2 generation worker")
    parser.add_argument("--control-dir", required=True)
    parser.add_argument("--config-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    control_dir = Path(args.control_dir).resolve()
    config = json.loads(args.config_json)
    os.chdir(ROOT)
    os.environ.setdefault("PYTHONPATH", ".")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    worker = GenerationWorker(control_dir, config)
    try:
        worker.load()
        worker.loop()
    except Exception as exc:
        worker.set_status("failed", error=str(exc), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
