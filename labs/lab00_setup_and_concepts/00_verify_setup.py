#!/usr/bin/env python
"""Lab 00: verify the full stack (transformers + vLLM + DeepSpeed) works on
this node before starting the distillation labs.

Run: python 00_verify_setup.py
"""
from __future__ import annotations

import time

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "In one sentence, what is knowledge distillation in machine learning?"


def section(title: str):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def check_gpus():
    section("1. GPU / CUDA / NCCL")
    import torch

    n = torch.cuda.device_count()
    assert n > 0, "No CUDA GPUs visible - are you on the right machine/env?"
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        print(f"  gpu{i}: {props.name}, {props.total_memory / 1024**3:.0f}GB")
    print(f"  bf16 supported: {torch.cuda.is_bf16_supported()}")
    import torch.distributed as dist

    print(f"  NCCL available: {dist.is_nccl_available()}")
    print("OK")


def check_transformers_generate():
    section("2. transformers: load + generate")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
    messages = [{"role": "user", "content": PROMPT}]
    inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
    t0 = time.time()
    out = model.generate(inputs["input_ids"], max_new_tokens=64, do_sample=False)
    dt = time.time() - t0
    text = tok.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    print(f"  generated in {dt:.1f}s: {text!r}")
    del model
    torch.cuda.empty_cache()
    print("OK")


def check_vllm_generate():
    section("3. vLLM: load + generate")
    from vllm import LLM, SamplingParams

    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.5, enforce_eager=True)
    tok = llm.get_tokenizer()
    prompt = tok.apply_chat_template([{"role": "user", "content": PROMPT}], add_generation_prompt=True, tokenize=False)

    t0 = time.time()
    outputs = llm.generate([prompt], SamplingParams(max_tokens=64, temperature=0.0))
    dt = time.time() - t0

    print(f"  generated in {dt:.1f}s: {outputs[0].outputs[0].text!r}")
    print("OK")


def check_deepspeed_flash_attn():
    section("4. DeepSpeed / flash-attn versions")
    import deepspeed

    print(f"  deepspeed {deepspeed.__version__}")
    try:
        import flash_attn

        print(f"  flash_attn {flash_attn.__version__}")
    except ImportError:
        print("  flash_attn not installed (optional - labs will fall back to sdpa attention)")
    print("OK")


def main():
    check_gpus()
    check_transformers_generate()
    check_vllm_generate()
    check_deepspeed_flash_attn()
    section("All checks passed - you're ready for lab01.")


if __name__ == "__main__":
    main()
