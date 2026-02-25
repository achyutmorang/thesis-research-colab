from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


LATENTDRIVER_CLONE_CMD = (
    "git clone --depth 1 https://github.com/Sephirex-X/LatentDriver.git "
    "/content/LatentDriver || true"
)
REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_COLAB_PATH = REPO_ROOT / "requirements-colab.txt"
NUMERIC_REPAIR_REQUIREMENTS = "numpy==2.2.6 scipy==1.14.1 pandas==2.2.3 scikit-learn==1.6.1"


def _sh(cmd: str) -> None:
    print(f"[setup] $ {cmd}")
    rc = subprocess.run(["bash", "-lc", cmd], check=False)
    if rc.returncode != 0:
        raise RuntimeError(f"Command failed ({rc.returncode}): {cmd}")


def _verify_repo_layout(repo_root: Path) -> None:
    closedloop_path = repo_root / "src" / "closedloop"
    legacy_trackb_path = repo_root / "src" / "trackb"
    if not closedloop_path.exists():
        raise RuntimeError(
            f"Expected module path missing: {closedloop_path}. "
            "You may be on a stale checkout; re-clone the repository."
        )
    if legacy_trackb_path.exists():
        raise RuntimeError(
            f"Legacy module path still present: {legacy_trackb_path}. "
            "Delete /content/thesis-research-colab and clone the latest main branch."
        )


def _probe_numpy_runtime() -> tuple[bool, str]:
    probe = subprocess.run(
        [
            "python",
            "-c",
            (
                "import numpy as np; "
                "from numpy._core.umath import _center, _expandtabs; "
                "print(np.__version__, np.__file__)"
            ),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if probe.returncode == 0:
        msg = probe.stdout.strip() or "NumPy probe succeeded."
        return True, msg
    err = (probe.stderr or probe.stdout).strip()
    return False, err


def _repair_numeric_stack() -> None:
    # Force-reinstall numeric stack to avoid mixed ABI/module states in Colab images.
    _sh("pip uninstall -y numpy scipy pandas scikit-learn || true")
    _sh(f"pip install --no-cache-dir --force-reinstall {NUMERIC_REPAIR_REQUIREMENTS}")


def _probe_core_runtime() -> tuple[bool, str]:
    probe = subprocess.run(
        [
            "python",
            "-c",
            (
                "import jax, waymax, numpy as np, pandas as pd, scipy, sklearn; "
                "from numpy._core.umath import _center, _expandtabs; "
                "print('ok', np.__version__, pd.__version__, scipy.__version__, sklearn.__version__, jax.__version__)"
            ),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if probe.returncode == 0:
        return True, (probe.stdout.strip() or "core runtime probe succeeded")
    return False, ((probe.stderr or probe.stdout).strip())


def _patch_transformers_import_block(repo_path: Path) -> None:
    gpt2_model_path = repo_path / "src/policy/latentdriver/gpt2_model.py"
    if not gpt2_model_path.exists():
        print(f"[patch] gpt2_model file missing, skipped: {gpt2_model_path}")
        return

    gpt2_txt = gpt2_model_path.read_text()
    old_block = """from transformers.modeling_utils import (
    Conv1D,
    PreTrainedModel,
    SequenceSummary,
    find_pruneable_heads_and_indices,
    prune_conv1d_layer,
)"""
    new_block = """from transformers.modeling_utils import PreTrainedModel

try:
    from transformers.modeling_utils import SequenceSummary
except Exception:
    class SequenceSummary(nn.Module):
        def __init__(self, config=None):
            super().__init__()
        def forward(self, hidden_states, *args, **kwargs):
            return hidden_states[:, -1]

try:
    from transformers.modeling_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
except Exception:
    from transformers.pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
"""
    if old_block in gpt2_txt:
        gpt2_model_path.write_text(gpt2_txt.replace(old_block, new_block))
        print("[patch] gpt2_model transformers compatibility patch applied")
    else:
        print("[patch] gpt2_model block already patched or changed")


def _patch_sort_vertices_fallback(repo_path: Path) -> None:
    sort_vert_path = repo_path / "src/ops/sort_vertices/sort_vert.py"
    if not sort_vert_path.exists():
        print(f"[patch] sort_vert file missing, skipped: {sort_vert_path}")
        return

    fallback_code = """import torch
from torch.autograd import Function

try:
    from src.ops.sort_vertices import sort_vertices as _sort_vertices_ext
except Exception:
    _sort_vertices_ext = None


def _sort_vertices_fallback(vertices, mask, num_valid):
    # vertices: [B,N,M,2], mask: [B,N,M], num_valid: [B,N]
    B, N, M, _ = vertices.shape
    out = torch.zeros((B, N, 9), dtype=torch.long, device=vertices.device)
    for b in range(B):
        for n in range(N):
            nv = int(num_valid[b, n].item())
            if nv <= 0:
                continue
            valid_idx = torch.nonzero(mask[b, n], as_tuple=False).flatten()
            if valid_idx.numel() == 0:
                continue
            valid_idx = valid_idx[:nv]
            pts = vertices[b, n, valid_idx]
            ang = torch.atan2(pts[:, 1], pts[:, 0])
            order = valid_idx[torch.argsort(ang)]
            k = min(int(order.numel()), 8)
            out[b, n, :k] = order[:k]
            out[b, n, k] = order[0]
            if k + 1 < 9:
                out[b, n, k + 1:] = order[k - 1]
    return out


class SortVertices(Function):
    @staticmethod
    def forward(ctx, vertices, mask, num_valid):
        if _sort_vertices_ext is not None:
            idx = _sort_vertices_ext.sort_vertices_forward(vertices, mask, num_valid)
        else:
            idx = _sort_vertices_fallback(vertices, mask, num_valid)
        ctx.mark_non_differentiable(idx)
        return idx

    @staticmethod
    def backward(ctx, gradout):
        return ()


sort_v = SortVertices.apply
"""
    sort_vert_path.write_text(fallback_code)
    print("[patch] sort_vert fallback patch applied")


def _ensure_checkpoint(ckpt_path: Path) -> None:
    if ckpt_path.exists():
        print("[ckpt] already present:", ckpt_path)
        return

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id="Sephirex-x/LatentDriver",
            filename="checkpoints/lantentdriver_t2_J3.ckpt",
            local_dir="/content",
        )
        print("[ckpt] downloaded:", downloaded)
    except Exception as e:
        print("[ckpt] hf_hub_download failed:", e)
        print("[ckpt] trying direct URL fallback...")
        rc = subprocess.run(
            [
                "bash",
                "-lc",
                "wget -O '/content/checkpoints/lantentdriver_t2_J3.ckpt' "
                "'https://huggingface.co/Sephirex-x/LatentDriver/resolve/main/checkpoints/lantentdriver_t2_J3.ckpt'",
            ],
            check=False,
        )
        if rc.returncode != 0:
            print("[ckpt] direct download failed; place checkpoint manually at", ckpt_path)

    if ckpt_path.exists():
        size_mb = round(ckpt_path.stat().st_size / (1024**2), 2)
        print("[ckpt] ready:", ckpt_path, "size_mb=", size_mb)
    else:
        print("[ckpt] missing:", ckpt_path)


def _install_requirements(requirements_path: Path) -> None:
    if not requirements_path.exists():
        raise FileNotFoundError(f"Missing requirements file: {requirements_path}")
    _sh("pip install --upgrade pip")
    _sh(f"pip install --upgrade -r '{requirements_path}'")


def _sync_latentdriver_repo() -> None:
    latentdriver_repo = Path("/content/LatentDriver")
    if latentdriver_repo.exists():
        print(f"[setup] LatentDriver repo already exists: {latentdriver_repo}")
        return
    _sh(LATENTDRIVER_CLONE_CMD)


def run_deterministic_setup(run_setup: bool = False, force_reinstall: bool = False) -> None:
    _verify_repo_layout(REPO_ROOT)

    if not run_setup:
        ok, details = _probe_numpy_runtime()
        if not ok:
            raise RuntimeError(
                "RUN_SETUP=False but runtime dependency probe failed. "
                "Set RUN_SETUP=True in this notebook cell, run setup once, restart runtime, "
                "then set RUN_SETUP=False and run all.\n"
                f"Probe error: {details}"
            )
        print(f"RUN_SETUP=False: runtime probe passed ({details}).")
        return

    print("[setup] Starting deterministic environment bootstrap")

    core_ok, core_details = _probe_core_runtime()
    if core_ok and (not force_reinstall):
        print(f"[setup] Core runtime already healthy; skipping heavy pip install ({core_details}).")
    else:
        if force_reinstall:
            print("[setup] force_reinstall=True -> running full dependency install.")
        else:
            print(f"[setup] Core runtime probe failed; installing dependencies.\n[setup] probe error: {core_details}")
        _install_requirements(REQUIREMENTS_COLAB_PATH)

    _sync_latentdriver_repo()

    latentdriver_repo = Path("/content/LatentDriver")
    if str(latentdriver_repo) not in sys.path:
        sys.path.insert(0, str(latentdriver_repo))
    os.environ["PYTHONPATH"] = str(latentdriver_repo) + ":" + os.environ.get("PYTHONPATH", "")

    _patch_transformers_import_block(latentdriver_repo)
    _patch_sort_vertices_fallback(latentdriver_repo)
    _ensure_checkpoint(Path("/content/checkpoints/lantentdriver_t2_J3.ckpt"))

    ok, details = _probe_numpy_runtime()
    if not ok:
        print("[setup] NumPy probe failed after lockfile install, repairing numeric stack...")
        _repair_numeric_stack()
        ok, details = _probe_numpy_runtime()
        if not ok:
            raise RuntimeError(
                "NumPy runtime probe failed after repair attempt. "
                "Restart runtime and re-run setup. "
                f"Probe error: {details}"
            )
    print(f"[setup] NumPy probe passed ({details}).")

    print("Setup complete. Restart runtime once, set RUN_SETUP=False, then Run all.")


if __name__ == "__main__":
    env_flag = os.environ.get("RUN_SETUP", "false").strip().lower()
    run_deterministic_setup(run_setup=env_flag in {"1", "true", "yes", "y"})
