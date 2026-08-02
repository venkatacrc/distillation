#!/usr/bin/env python
"""Top-level orchestrator: runs some or all of labs 00-09 end-to-end,
executing each lab's own scripts in the order documented in its README.

This is a convenience runner for working through (or re-running) the whole
curriculum unattended - it is NOT a substitute for reading each lab's
README first. Several labs are long (lab05/06/08/09 can take 1-4 hours)
and use most of an 8-GPU node; read the top-level README's hardware notes
and use --dry-run to see the full command plan before committing to a
full run.

Examples
--------
    # print every command that would run, without running anything
    python scripts/run_all_labs.py --dry-run

    # run just the fast, single-GPU fundamentals labs
    python scripts/run_all_labs.py --labs lab00 lab01 lab02

    # run everything, using up to 8 GPUs for the multi-GPU steps
    python scripts/run_all_labs.py --num-gpus 8

    # resume a full run starting at lab05, pushing through any failures
    python scripts/run_all_labs.py --from-lab lab05 --continue-on-error

    # keep per-step logs on disk instead of streaming everything to one terminal
    python scripts/run_all_labs.py --log-dir logs/run_all_labs
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LABS_DIR = REPO_ROOT / "labs"

ALL_LABS = [
    "lab00_setup_and_concepts",
    "lab01_classic_kd_hinton",
    "lab02_kd_loss_landscape",
    "lab03_response_based_sft_distillation",
    "lab04_onpolicy_gkd",
    "lab05_prune_and_distill",
    "lab06_reasoning_cot_distillation",
    "lab07_distillation_vs_rl_ablation",
    "lab08_scaled_pipeline_multi_gpu",
    "lab09_capstone_full_node",
]
SHORT_TO_FULL = {name.split("_")[0]: name for name in ALL_LABS}  # "lab00" -> "lab00_setup_and_concepts"


@dataclass
class Step:
    description: str
    cmd: list[str]


def normalize_lab(name: str) -> str:
    name = name.strip().rstrip("/")
    if name in ALL_LABS:
        return name
    short = name.split("_")[0]
    if short in SHORT_TO_FULL:
        return SHORT_TO_FULL[short]
    raise argparse.ArgumentTypeError(
        f"Unknown lab {name!r}. Use a short id ({', '.join(SHORT_TO_FULL)}) or a full directory name."
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labs", nargs="+", type=normalize_lab, default=None, help="specific labs to run, e.g. lab00 lab03 (default: all)")
    p.add_argument("--from-lab", type=normalize_lab, default=None, help="start the full sequence at this lab (ignored if --labs is set)")
    p.add_argument("--to-lab", type=normalize_lab, default=None, help="stop the full sequence after this lab, inclusive (ignored if --labs is set)")
    p.add_argument("--num-gpus", type=int, default=None, help="GPU count for multi-GPU steps (default: auto-detect via nvidia-smi)")
    p.add_argument("--teacher-gpus", default="0,1", help="CUDA device ids reserved for lab08's teacher server (default: 0,1)")
    p.add_argument("--dry-run", action="store_true", help="print the commands that would run without executing them")
    p.add_argument("--continue-on-error", action="store_true", help="keep going through remaining steps/labs after a failure")
    p.add_argument("--skip-install", action="store_true", help="skip the one-time `pip install -e .` step")
    p.add_argument("--skip-env-check", action="store_true", help="skip running scripts/check_env.py before starting")
    p.add_argument("--log-dir", default=None, help="write each step's stdout/stderr under this directory instead of streaming to the console")
    return p.parse_args()


def select_labs(args: argparse.Namespace) -> list[str]:
    if args.labs:
        return args.labs
    start = ALL_LABS.index(args.from_lab) if args.from_lab else 0
    end = ALL_LABS.index(args.to_lab) + 1 if args.to_lab else len(ALL_LABS)
    return ALL_LABS[start:end]


def detect_gpu_count() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
        n = len([line for line in out.stdout.splitlines() if line.strip()])
        return max(n, 1)
    except Exception:
        return 1


def accelerate_launch_prefix(n: int) -> list[str]:
    """Multi-process launcher for lab05's Accelerate-based script; falls
    back to plain `python` for a single-GPU/CPU run (Accelerator() works
    fine unlaunched in that case)."""
    if n <= 1:
        return [sys.executable]
    return ["accelerate", "launch", "--multi_gpu", "--num_processes", str(n)]


def slugify(text: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in text.lower()).strip("-")


def run_step(step: Step, cwd: Path, dry_run: bool, log_dir: Path | None) -> tuple[bool, float]:
    print(f"\n{'>' * 70}\n>> [{cwd.name}] {step.description}\n>> {' '.join(step.cmd)}\n{'>' * 70}")
    if dry_run:
        return True, 0.0

    t0 = time.time()
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{slugify(step.description)}.log"
        with open(log_path, "w") as f:
            result = subprocess.run(step.cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
        print(f"   (output logged to {log_path})")
    else:
        result = subprocess.run(step.cmd, cwd=cwd)
    elapsed = time.time() - t0
    ok = result.returncode == 0
    print(f">> {'OK' if ok else f'FAILED (exit {result.returncode})'} in {elapsed:.0f}s")
    return ok, elapsed


def build_steps(lab: str, num_gpus: int) -> list[Step]:
    py = sys.executable
    if lab == "lab00_setup_and_concepts":
        return [Step("verify environment", [py, "00_verify_setup.py"])]

    if lab == "lab01_classic_kd_hinton":
        return [
            Step("fine-tune teacher + cache logits", [py, "train_teacher.py"]),
            Step("train baseline student", [py, "train_student_baseline.py"]),
            Step("train KD student sweep", [py, "train_student_kd.py"]),
            Step("plot results", [py, "plot_results.py"]),
        ]

    if lab == "lab02_kd_loss_landscape":
        return [
            Step("toy forward/reverse KL demo", [py, "toy_distributions.py"]),
            Step("real token-level divergences", [py, "token_level_kd_losses.py"]),
        ]

    if lab == "lab03_response_based_sft_distillation":
        return [
            Step("generate teacher responses", [py, "generate_teacher_responses.py"]),
            Step("SFT the student", [py, "train_student_sft.py"]),
            Step("evaluate (judge win-rate)", [py, "evaluate.py"]),
        ]

    if lab == "lab04_onpolicy_gkd":
        return [
            Step("train GKD sweep", [py, "train_gkd.py"]),
            Step("compare vs off-policy baseline", [py, "compare_offpolicy_vs_onpolicy.py"]),
        ]

    if lab == "lab05_prune_and_distill":
        n = min(num_gpus, 4)
        return [
            Step("compute layer importance", [py, "compute_layer_importance.py"]),
            Step("prune layers", [py, "prune_depth.py"]),
            Step(f"distillation recovery ({n} GPU{'s' if n != 1 else ''})", accelerate_launch_prefix(n) + ["distill_recover.py"]),
            Step("evaluate compression", [py, "evaluate_compression.py"]),
        ]

    if lab == "lab06_reasoning_cot_distillation":
        return [
            Step("generate CoT traces (rejection sampling)", [py, "generate_traces.py"]),
            Step("filter traces", [py, "filter_traces.py"]),
            Step("SFT the student on filtered traces", [py, "train_student_cot_sft.py"]),
            Step("evaluate GSM8K + coverage", [py, "evaluate_gsm8k.py"]),
        ]

    if lab == "lab07_distillation_vs_rl_ablation":
        return [
            Step("train GRPO baseline", [py, "train_grpo_baseline.py"]),
            Step("compare distillation vs RL", [py, "compare_distillation_vs_rl.py"]),
        ]

    if lab == "lab09_capstone_full_node":
        return [
            Step("generate teacher data (Alpaca + GSM8K)", [py, "01_generate_teacher_data.py"]),
            Step("SFT the student", [py, "02_sft_student.py"]),
            Step("on-policy GKD refinement", [py, "03_gkd_refine.py"]),
            Step("run eval suite", [py, "eval_suite.py"]),
            Step("generate report", [py, "report.py"]),
        ]

    if lab == "lab08_scaled_pipeline_multi_gpu":
        return []  # handled specially by run_lab08(), which needs a background server

    raise ValueError(f"No steps registered for lab {lab!r}")


def wait_for_server(url: str, timeout_s: int) -> bool:
    import requests

    print(f"Waiting for teacher server at {url} (timeout {timeout_s // 60} min)...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            if requests.get(url, timeout=5).status_code == 200:
                print(f"Server ready after {time.time() - t0:.0f}s")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(10)
    return False


def start_teacher_server(lab_dir: Path, model: str, gpus: str, port: int, log_dir: Path | None) -> subprocess.Popen:
    cmd = ["bash", "serve_teacher_vllm.sh", model, gpus, str(port)]
    print(f"\n{'>' * 70}\n>> [{lab_dir.name}] starting teacher server in background\n>> {' '.join(cmd)}\n{'>' * 70}")
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout = open(log_dir / "serve_teacher_vllm.log", "w")
    else:
        stdout = subprocess.DEVNULL
    return subprocess.Popen(cmd, cwd=lab_dir, stdout=stdout, stderr=subprocess.STDOUT)


def stop_teacher_server(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    print("Stopping teacher server...")
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def run_lab08(args: argparse.Namespace, lab_dir: Path, log_dir: Path | None) -> list[tuple[str, bool, float]]:
    """lab08 needs a background vLLM server (started/stopped twice - once
    for data generation + SFT, once for the final evaluation - and shut
    down in between so the on-policy GKD step has room to load the teacher
    in-process), so it can't use the generic sequential-steps runner."""
    import yaml

    cfg = yaml.safe_load(open(lab_dir / "config.yaml"))
    total_gpus = args.num_gpus if args.num_gpus is not None else detect_gpu_count()
    teacher_gpu_ids = [g.strip() for g in args.teacher_gpus.split(",") if g.strip()]
    student_gpu_ids = [str(i) for i in range(total_gpus) if str(i) not in teacher_gpu_ids]

    if total_gpus < len(teacher_gpu_ids) + 1:
        print(
            f"Skipping lab08: needs >={len(teacher_gpu_ids) + 1} GPUs "
            f"({len(teacher_gpu_ids)} for the teacher server + >=1 for the student), only {total_gpus} available."
        )
        return []

    port = cfg["teacher_server"]["port"]
    results: list[tuple[str, bool, float]] = []

    def run_and_record(step: Step) -> bool:
        ok, dt = run_step(step, lab_dir, args.dry_run, log_dir)
        results.append((step.description, ok, dt))
        return ok

    py = sys.executable

    proc = None
    try:
        proc = None if args.dry_run else start_teacher_server(lab_dir, cfg["teacher_model"], args.teacher_gpus, port, log_dir)
        if not args.dry_run and not wait_for_server(f"http://localhost:{port}/v1/models", timeout_s=1800):
            raise RuntimeError("Teacher server did not become ready in time")

        if not run_and_record(Step("generate teacher responses via API", [py, "generate_teacher_responses_api.py"])):
            if not args.continue_on_error:
                return results

        n_student = max(len(student_gpu_ids), 1)
        ds_cmd = ["deepspeed", "--num_gpus", str(n_student)]
        if student_gpu_ids:
            ds_cmd += ["--include", f"localhost:{','.join(student_gpu_ids)}"]
        ds_cmd += ["train_student_deepspeed.py"]
        if not run_and_record(Step(f"SFT via DeepSpeed ZeRO-3 ({n_student} GPUs)", ds_cmd)):
            if not args.continue_on_error:
                return results
    finally:
        stop_teacher_server(proc)

    if not run_and_record(Step("on-policy GKD refinement (teacher loaded in-process)", [py, "train_gkd_scaled.py"])):
        if not args.continue_on_error:
            return results

    proc2 = None
    try:
        proc2 = None if args.dry_run else start_teacher_server(lab_dir, cfg["teacher_model"], args.teacher_gpus, port, log_dir)
        if not args.dry_run and not wait_for_server(f"http://localhost:{port}/v1/models", timeout_s=1800):
            raise RuntimeError("Teacher server did not become ready in time")
        run_and_record(Step("evaluate scaled students", [py, "evaluate_scaled.py"]))
    finally:
        stop_teacher_server(proc2)

    return results


def main() -> None:
    args = parse_args()
    labs = select_labs(args)
    log_dir_root = Path(args.log_dir).resolve() if args.log_dir else None

    print(f"Repo root: {REPO_ROOT}")
    print(f"Labs to run, in order: {', '.join(labs)}")
    if args.dry_run:
        print("(--dry-run: no commands will actually be executed)")

    if not args.skip_install:
        print("\nEnsuring the `common` package is installed (pip install -e .)...")
        if not args.dry_run:
            subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], cwd=REPO_ROOT, check=True)

    if not args.skip_env_check:
        ok, _ = run_step(
            Step("environment check", [sys.executable, "scripts/check_env.py"]),
            REPO_ROOT,
            args.dry_run,
            (log_dir_root / "_env_check") if log_dir_root else None,
        )
        if not ok and not args.continue_on_error:
            print("\nEnvironment check failed. Fix the issues above, or re-run with --skip-env-check / --continue-on-error.")
            sys.exit(1)

    num_gpus = args.num_gpus if args.num_gpus is not None else detect_gpu_count()
    print(f"Using num_gpus={num_gpus} for multi-GPU steps (override with --num-gpus)")

    all_results: list[tuple[str, str, bool, float]] = []
    t_start = time.time()

    for lab in labs:
        lab_dir = LABS_DIR / lab
        lab_log_dir = (log_dir_root / lab) if log_dir_root else None
        print(f"\n{'#' * 70}\n# {lab}\n{'#' * 70}")

        if lab == "lab08_scaled_pipeline_multi_gpu":
            lab_results = run_lab08(args, lab_dir, lab_log_dir)
        else:
            lab_results = []
            for step in build_steps(lab, num_gpus):
                ok, dt = run_step(step, lab_dir, args.dry_run, lab_log_dir)
                lab_results.append((step.description, ok, dt))
                if not ok and not args.continue_on_error:
                    break

        all_results.extend((lab, desc, ok, dt) for desc, ok, dt in lab_results)

        if any(not ok for _, ok, _ in lab_results) and not args.continue_on_error:
            print(f"\nStopping: a step in {lab} failed. Re-run with --continue-on-error to push through failures.")
            break

    total_elapsed = time.time() - t_start
    print(f"\n{'=' * 70}\nSummary ({total_elapsed / 60:.1f} min total)\n{'=' * 70}")
    for lab, desc, ok, dt in all_results:
        print(f"  [{'OK  ' if ok else 'FAIL'}] {lab:<40} {desc:<45} {dt:6.0f}s")

    n_failed = sum(1 for _, _, ok, _ in all_results if not ok)
    if n_failed:
        print(f"\n{n_failed} step(s) failed.")
        sys.exit(1)
    print("\nAll requested steps completed successfully.")


if __name__ == "__main__":
    main()
