#!/usr/bin/env python
"""Environment sanity check for the distillation labs.

Verifies GPUs, CUDA/NCCL, bf16/tf32 support, required Python packages, the
HF cache location, and available disk space. Mirrors the PASS/FAIL style of
the diagnostic most labs on this kind of node already use, so you can run
it the same way:

    python scripts/check_env.py

Exits with a non-zero status if any check marked "required" fails.
"""
from __future__ import annotations

import importlib
import os
import shutil
import sys

MIN_VERSIONS = {
    "transformers": "4.47",
    "accelerate": "1.2",
    "trl": "0.27",
    "peft": "0.14",
    "datasets": "3.2",
    "deepspeed": "0.16",
    "vllm": "0.7",
}

RESULTS: list[tuple[bool, str, str, bool]] = []  # (passed, label, detail, required)


def check(label: str, required: bool = True):
    def decorator(fn):
        try:
            ok, detail = fn()
        except Exception as exc:  # noqa: BLE001 - want to report any failure
            ok, detail = False, f"raised {type(exc).__name__}: {exc}"
        RESULTS.append((ok, label, detail, required))
        return fn

    return decorator


def _version_ge(installed: str, minimum: str) -> bool:
    def parts(v):
        v = v.split("+")[0].split("-")[0]
        return [int(p) for p in v.split(".") if p.isdigit()]

    return parts(installed) >= parts(minimum)


@check("python version")
def _python_version():
    v = sys.version.split()[0]
    return v >= "3.10", v


@check("torch import + CUDA build")
def _torch_build():
    import torch

    return True, f"torch {torch.__version__}, CUDA build {torch.version.cuda}"


@check("CUDA available")
def _cuda_available():
    import torch

    n = torch.cuda.device_count()
    return torch.cuda.is_available() and n > 0, f"{n} device(s) visible"


@check("GPU inventory")
def _gpu_inventory():
    import torch

    if not torch.cuda.is_available():
        return False, "no CUDA devices"
    infos = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / (1024**3)
        infos.append(f"gpu{i}={props.name} ({mem_gb:.0f}GB, sm_{props.major}{props.minor})")
    return True, "; ".join(infos)


@check("bf16 / tf32 support")
def _bf16_tf32():
    import torch

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.backends.cuda.matmul.allow_tf32 = True
    return bf16, f"bf16_supported={bf16}, tf32 enabled for matmul"


@check("NCCL backend")
def _nccl():
    import torch.distributed as dist

    return dist.is_nccl_available(), "torch.distributed.is_nccl_available()"


@check("flash-attention (optional, speeds up labs 03/04/06/07/08/09)", required=False)
def _flash_attn():
    import flash_attn

    return True, f"flash_attn {flash_attn.__version__}"


def _make_package_check(name: str, min_version: str | None):
    @check(f"import {name}" + (f" (expect >= {min_version})" if min_version else ""))
    def _inner():
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "unknown")
        if min_version and version != "unknown":
            return _version_ge(version, min_version), f"{name} {version}"
        return True, f"{name} {version}"

    return _inner


for pkg, min_v in MIN_VERSIONS.items():
    _make_package_check(pkg, min_v)


@check("HF cache location (should be on a large scratch volume, not root disk)", required=False)
def _hf_cache():
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    on_root = os.path.abspath(hf_home).startswith(("/home", "/root")) and "/raid" not in hf_home
    return not on_root, hf_home


@check("disk space on cache volume")
def _disk_space():
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    os.makedirs(hf_home, exist_ok=True)
    total, used, free = shutil.disk_usage(hf_home)
    free_gb = free / (1024**3)
    return free_gb > 200, f"{free_gb:.0f}GB free at {hf_home}"


def main() -> int:
    label_width = max(len(label) for _, label, _, _ in RESULTS)
    any_required_failed = False
    print(f"Environment check ({len(RESULTS)} checks)\n")
    for ok, label, detail, required in RESULTS:
        tag = "PASS" if ok else ("FAIL" if required else "WARN")
        if not ok and required:
            any_required_failed = True
        print(f"  [{tag}] {label.ljust(label_width)}  {detail}")

    print()
    if any_required_failed:
        print("One or more REQUIRED checks failed. Fix these before running the labs.")
        return 1
    print("All required checks passed. Optional (WARN) items are nice-to-have.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
