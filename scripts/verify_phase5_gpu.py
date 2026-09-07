"""Verify that the GPU can actually execute training operations.

    python scripts/verify_phase5_gpu.py

This is a hard gate for Phase 5. It does not accept `nvidia-smi` working as proof: it runs
real CUDA tensor maths, a real model forward pass on the GPU, a real backward pass, and an
optimizer step, then reports measured VRAM use.

Exits non-zero if GPU execution is unavailable, so callers cannot mistake a CPU fallback
for GPU training. Writes ``scripts/phase5_gpu_report.txt``.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REPORT = Path(__file__).resolve().parent / "phase5_gpu_report.txt"


def emit(message: str = "") -> None:
    print(message, flush=True)
    with REPORT.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def package_version(name: str) -> str:
    try:
        module = __import__(name)
        return getattr(module, "__version__", "installed")
    except Exception:  # noqa: BLE001
        return "NOT INSTALLED"


def main() -> int:
    REPORT.unlink(missing_ok=True)

    emit("=" * 74)
    emit("ENVIRONMENT")
    emit("=" * 74)
    emit(f"  OS              : {platform.platform()}")
    emit(f"  Python          : {sys.version.split()[0]}")
    for name in ("torch", "torchvision", "transformers", "peft", "accelerate", "numpy", "PIL"):
        emit(f"  {name:15s} : {package_version(name)}")

    emit()
    emit("=" * 74)
    emit("nvidia-smi")
    emit("=" * 74)
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        emit(f"  {output.stdout.strip() or output.stderr.strip()}")
    except Exception as exc:  # noqa: BLE001
        emit(f"  nvidia-smi unavailable: {exc}")

    import torch

    emit()
    emit("=" * 74)
    emit("TORCH CUDA")
    emit("=" * 74)
    emit(f"  torch.__version__        : {torch.__version__}")
    emit(f"  torch.version.cuda       : {torch.version.cuda}")
    emit(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        emit()
        emit("RESULT: FAILED - CUDA is not available to PyTorch.")
        emit("Phase 5 GPU training cannot proceed.")
        return 1

    device = torch.device("cuda:0")
    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory
    emit(f"  device name              : {name}")
    emit(f"  compute capability       : {capability[0]}.{capability[1]}")
    emit(f"  total VRAM               : {total_vram / 1024 ** 3:.2f} GB")

    # --- 1. real CUDA tensor maths, verified numerically --------------------------
    emit()
    emit("=" * 74)
    emit("1. CUDA TENSOR OPERATION")
    emit("=" * 74)
    a = torch.randn(512, 512, device=device)
    b = torch.randn(512, 512, device=device)
    product = a @ b
    torch.cuda.synchronize()
    # Verify against CPU to prove the GPU actually computed the right answer.
    expected = a.cpu() @ b.cpu()
    max_error = (product.cpu() - expected).abs().max().item()
    emit(f"  matmul on {product.device} -> {tuple(product.shape)}")
    emit(f"  max abs difference vs CPU : {max_error:.2e}")
    if max_error > 1e-2:
        emit("RESULT: FAILED - GPU maths does not agree with CPU.")
        return 1
    emit("  CUDA tensor operation VERIFIED")

    # --- 2. real model forward + backward + optimizer step -----------------------
    emit()
    emit("=" * 74)
    emit("2. MODEL FORWARD / BACKWARD / OPTIMIZER ON GPU")
    emit("=" * 74)
    torch.cuda.reset_peak_memory_stats(device)

    model = torch.nn.Sequential(
        torch.nn.Linear(256, 512),
        torch.nn.GELU(),
        torch.nn.Linear(512, 128),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    inputs = torch.randn(32, 256, device=device)
    targets = torch.randn(32, 128, device=device)

    before = [p.detach().clone() for p in model.parameters()]

    outputs = model(inputs)
    loss = torch.nn.functional.mse_loss(outputs, targets)
    emit(f"  forward  : output {tuple(outputs.shape)} on {outputs.device}")
    emit(f"  loss     : {loss.item():.6f}")

    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    emit(f"  backward : total grad norm {grad_norm:.6f}")
    if grad_norm == 0.0:
        emit("RESULT: FAILED - gradients are all zero.")
        return 1

    optimizer.step()
    torch.cuda.synchronize()
    changed = sum(
        1 for old, new in zip(before, model.parameters(), strict=True)
        if not torch.equal(old, new.detach())
    )
    emit(f"  optimizer: {changed}/{len(before)} parameter tensors updated")
    if changed == 0:
        emit("RESULT: FAILED - optimizer did not update any parameters.")
        return 1

    peak = torch.cuda.max_memory_allocated(device)
    emit(f"  peak VRAM allocated      : {peak / 1024 ** 2:.1f} MB")

    # --- 3. mixed precision, used by the training run ----------------------------
    emit()
    emit("=" * 74)
    emit("3. MIXED PRECISION (fp16 autocast)")
    emit("=" * 74)
    try:
        scaler = torch.amp.GradScaler("cuda")
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            loss = torch.nn.functional.mse_loss(model(inputs), targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        emit(f"  fp16 autocast step OK (loss {loss.item():.6f})")
    except Exception as exc:  # noqa: BLE001
        emit(f"  fp16 autocast FAILED: {type(exc).__name__}: {exc}")
        emit("  (training will fall back to fp32)")

    emit()
    emit("=" * 74)
    emit("RESULT: GPU VERIFIED - real CUDA forward, backward, and optimizer step executed.")
    emit("=" * 74)
    emit(f"  GPU  : {name}")
    emit(f"  VRAM : {total_vram / 1024 ** 3:.2f} GB")
    emit(f"  CUDA : {torch.version.cuda}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
