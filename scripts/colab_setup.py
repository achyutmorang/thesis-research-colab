from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


WAYMAX_INSTALL_CMD = (
    "pip install --upgrade "
    "'git+https://github.com/waymo-research/waymax.git@main#egg=waymo-waymax'"
)
JAX_CUDA12_INSTALL_CMD = (
    "pip install --upgrade "
    "'jax[cuda12]>=0.7.0,<0.8' "
    "-f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
)
LATENTDRIVER_DEP_INSTALL_CMD = (
    "pip install --upgrade "
    "hydra-core==1.3.2 omegaconf==2.3.0 lightning==2.3.3 "
    "pytorch-lightning==2.3.3 einops==0.8.0 "
    "transformers==4.46.3 huggingface_hub"
)
LATENTDRIVER_CLONE_CMD = (
    "git clone --depth 1 https://github.com/Sephirex-X/LatentDriver.git "
    "/content/LatentDriver || true"
)


def _sh(cmd: str) -> None:
    print(f"[setup] $ {cmd}")
    rc = subprocess.run(["bash", "-lc", cmd], check=False)
    if rc.returncode != 0:
        raise RuntimeError(f"Command failed ({rc.returncode}): {cmd}")


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


def run_deterministic_setup(run_setup: bool = False) -> None:
    if not run_setup:
        print("RUN_SETUP=False: skipping dependency setup.")
        return

    print("[setup] Starting deterministic environment bootstrap")

    _sh("pip install --upgrade pip")
    _sh(WAYMAX_INSTALL_CMD)
    _sh(JAX_CUDA12_INSTALL_CMD)
    _sh("pip install --upgrade mediapy seaborn scikit-learn tqdm")
    _sh(LATENTDRIVER_DEP_INSTALL_CMD)
    _sh(LATENTDRIVER_CLONE_CMD)

    latentdriver_repo = Path("/content/LatentDriver")
    if str(latentdriver_repo) not in sys.path:
        sys.path.insert(0, str(latentdriver_repo))
    os.environ["PYTHONPATH"] = str(latentdriver_repo) + ":" + os.environ.get("PYTHONPATH", "")

    _patch_transformers_import_block(latentdriver_repo)
    _patch_sort_vertices_fallback(latentdriver_repo)
    _ensure_checkpoint(Path("/content/checkpoints/lantentdriver_t2_J3.ckpt"))

    print("Setup complete. Restart runtime once, set RUN_SETUP=False, then Run all.")


if __name__ == "__main__":
    env_flag = os.environ.get("RUN_SETUP", "false").strip().lower()
    run_deterministic_setup(run_setup=env_flag in {"1", "true", "yes", "y"})
