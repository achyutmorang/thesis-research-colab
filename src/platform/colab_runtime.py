from __future__ import annotations

import importlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple


@dataclass
class RepoSyncResult:
    repo_dir: str
    repo_rev: str
    sys_path_head: Tuple[str, ...] = field(default_factory=tuple)


@dataclass
class DriveReadyResult:
    is_colab: bool
    mounted: bool
    required_folder: str
    writable: bool
    detail: str = ""


@dataclass
class SetupResult:
    result: Dict[str, Any]
    cache_hit: bool
    cache_reason: str
    cache_path: str


@dataclass
class RuntimeBootstrapResult:
    repo_sync: RepoSyncResult
    drive_status: DriveReadyResult
    setup: SetupResult
    prepared_repo_dir: str


@dataclass
class ColabRuntimeConfig:
    repo_url: str
    repo_dir: str = "/content/waymax-simulation-experiments"
    repo_branch: str = "main"
    required_drive_folder: str = "/content/drive/MyDrive/waymax_experiments"
    verify_drive_access_every_run: bool = False
    force_reinstall: bool = False
    auto_restart_after_setup: bool = True
    strict_lockfile_check: bool = True
    setup_cache_enabled: bool = True
    revalidate_core_imports_on_cache_hit: bool = True
    setup_cache_path: str = "/content/.closedloop_setup_cache.json"
    force_module_hot_reload: bool = True

    def to_bootstrap_kwargs(self) -> Dict[str, Any]:
        return {
            "repo_url": self.repo_url,
            "repo_dir": self.repo_dir,
            "repo_branch": self.repo_branch,
            "required_drive_folder": self.required_drive_folder,
            "verify_drive_access_every_run": bool(self.verify_drive_access_every_run),
            "force_reinstall": bool(self.force_reinstall),
            "auto_restart_after_setup": bool(self.auto_restart_after_setup),
            "strict_lockfile_check": bool(self.strict_lockfile_check),
            "setup_cache_enabled": bool(self.setup_cache_enabled),
            "revalidate_core_imports_on_cache_hit": bool(self.revalidate_core_imports_on_cache_hit),
            "setup_cache_path": self.setup_cache_path,
            "force_module_hot_reload": bool(self.force_module_hot_reload),
        }


_DRIVE_READY_CACHE = False


def _run(*args: str) -> None:
    subprocess.run(list(args), check=True)


def ensure_repo_checkout(repo_url: str, repo_dir: str, branch: str = "main") -> RepoSyncResult:
    repo_path = Path(repo_dir)
    content_root = Path("/content")
    content_root.mkdir(parents=True, exist_ok=True)

    try:
        _ = os.getcwd()
    except FileNotFoundError:
        os.chdir(str(content_root))

    if repo_path.exists() and (Path.cwd().resolve() == repo_path.resolve()):
        os.chdir(str(content_root))

    if repo_path.exists():
        print(f"[repo] existing checkout: {repo_path}")
        _run("git", "-C", str(repo_path), "fetch", "origin")
        _run("git", "-C", str(repo_path), "checkout", branch)
        _run("git", "-C", str(repo_path), "pull", "--ff-only", "origin", branch)
    else:
        print(f"[repo] cloning {repo_url} -> {repo_path}")
        _run("git", "clone", "--depth", "1", "-b", branch, repo_url, str(repo_path))

    if not (repo_path / "src" / "closedloop").exists():
        raise RuntimeError(f"[repo] missing expected module path: {repo_path / 'src' / 'closedloop'}")

    os.chdir(str(repo_path))
    repo_root = str(repo_path)
    src_root = str(repo_path / "src")
    for path in (repo_root, src_root):
        if path not in sys.path:
            sys.path.insert(0, path)

    rev = subprocess.check_output(["git", "-C", str(repo_path), "rev-parse", "--short", "HEAD"], text=True).strip()
    return RepoSyncResult(
        repo_dir=repo_root,
        repo_rev=str(rev),
        sys_path_head=tuple(sys.path[:5]),
    )


def _cleanup_stale_drive_mount_state() -> None:
    subprocess.run(["bash", "-lc", "fusermount -u /content/drive || true"], check=False)
    subprocess.run(["bash", "-lc", "umount /content/drive || true"], check=False)
    os.makedirs("/content/drive", exist_ok=True)


def _mount_drive_with_retries() -> None:
    from google.colab import auth, drive

    if os.path.ismount("/content/drive"):
        print("[drive] /content/drive already mounted")
        return

    errors = []
    attempts = [
        ("initial_mount", False, False, False),
        ("force_remount", True, False, False),
        ("auth_and_cleanup_force", True, True, True),
    ]
    for label, force_remount, do_auth, do_cleanup in attempts:
        try:
            if do_auth:
                auth.authenticate_user()
            if do_cleanup:
                _cleanup_stale_drive_mount_state()
            drive.mount("/content/drive", force_remount=force_remount)
            if os.path.ismount("/content/drive"):
                print(f"[drive] mount succeeded via {label}")
                return
            errors.append(f"{label}: mount returned without active mount")
        except Exception as e:
            errors.append(f"{label}: {type(e).__name__}: {e}")
            time.sleep(1.0)

    raise RuntimeError(
        "[drive] mount failed after retries. "
        + " | ".join(errors)
        + " ; If this persists: Runtime -> Disconnect and delete runtime, then rerun Step 2."
    )


def ensure_drive_ready(
    required_drive_folder: str = "/content/drive/MyDrive/waymax_experiments",
    verify_drive_access_every_run: bool = False,
) -> DriveReadyResult:
    global _DRIVE_READY_CACHE

    required = Path(required_drive_folder)
    is_colab = bool(os.environ.get("COLAB_RELEASE_TAG"))
    if not is_colab:
        return DriveReadyResult(
            is_colab=False,
            mounted=False,
            required_folder=str(required),
            writable=False,
            detail="non-colab-runtime",
        )

    if (
        _DRIVE_READY_CACHE
        and os.path.ismount("/content/drive")
        and required.exists()
        and (not bool(verify_drive_access_every_run))
    ):
        print("[drive] already validated in this runtime; skipping remount/probe.")
        return DriveReadyResult(
            is_colab=True,
            mounted=True,
            required_folder=str(required),
            writable=True,
            detail="cached_ready",
        )

    _mount_drive_with_retries()

    if not required.exists():
        raise RuntimeError(
            "[drive] Required folder missing: /content/drive/MyDrive/waymax_experiments. "
            "Ask Achyut Morang (owner) for edit access, then rerun this cell."
        )

    probe_file = required / f".codex_write_probe_{int(time.time() * 1e6)}.tmp"
    try:
        probe_file.write_text("ok", encoding="utf-8")
        probe_file.unlink(missing_ok=True)
    except Exception as e:
        raise RuntimeError(
            "[drive] Folder exists but is not writable: /content/drive/MyDrive/waymax_experiments. "
            "Ask Achyut Morang (owner) for edit access, then rerun this cell."
        ) from e

    _DRIVE_READY_CACHE = True
    print(f"[drive] Verified read/write access: {required}")
    return DriveReadyResult(
        is_colab=True,
        mounted=True,
        required_folder=str(required),
        writable=True,
        detail="ready",
    )


def _setup_fingerprint(requirements_path: Path, strict_lockfile_check: bool) -> str:
    import hashlib

    h = hashlib.sha256()
    h.update(requirements_path.read_bytes())
    h.update(str(sys.executable).encode("utf-8"))
    h.update(str(sys.version).encode("utf-8"))
    h.update(str(bool(strict_lockfile_check)).encode("utf-8"))
    return h.hexdigest()


def _core_import_probe() -> Tuple[bool, str]:
    try:
        import jax  # noqa: F401
        import numpy as np  # noqa: F401
        import pandas as pd  # noqa: F401
        import scipy  # noqa: F401
        import sklearn  # noqa: F401
        import waymax  # noqa: F401
        from numpy._core.umath import _center, _expandtabs  # noqa: F401

        return True, f"ok numpy={np.__version__} pandas={pd.__version__}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def run_cached_deterministic_setup(
    repo_root: str = ".",
    force_reinstall: bool = False,
    auto_restart_after_setup: bool = True,
    strict_lockfile_check: bool = True,
    setup_cache_enabled: bool = True,
    revalidate_core_imports_on_cache_hit: bool = True,
    setup_cache_path: str = "/content/.closedloop_setup_cache.json",
    repo_rev: str = "unknown",
) -> SetupResult:
    repo_root_path = Path(repo_root)
    setup_py = repo_root_path / "scripts" / "colab_setup.py"
    req_path = repo_root_path / "requirements-colab.txt"
    cache_path = Path(setup_cache_path)

    if not setup_py.exists():
        raise FileNotFoundError(f"Setup helper missing: {setup_py}")
    if not req_path.exists():
        raise FileNotFoundError(f"Requirements file missing: {req_path}")

    spec = importlib.util.spec_from_file_location("colab_setup", str(setup_py))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load setup helper from {setup_py}")
    colab_setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(colab_setup)

    fingerprint = _setup_fingerprint(req_path, strict_lockfile_check)
    cached: Dict[str, Any] = {}
    cache_hit = False
    cache_reason = ""
    setup_result: Dict[str, Any] | None = None

    if setup_cache_enabled and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
        except Exception:
            cached = {}

    if setup_cache_enabled and (not force_reinstall):
        if cached.get("fingerprint") == fingerprint and cached.get("status") == "ready":
            cache_hit = True
            cache_reason = "fingerprint_match"

    if cache_hit and revalidate_core_imports_on_cache_hit:
        core_ok, core_msg = _core_import_probe()
        if not core_ok:
            cache_hit = False
            cache_reason = f"cache_invalid_core_probe_failed: {core_msg}"
        else:
            setup_result = {
                "ran_setup": False,
                "cache_hit": True,
                "cache_reason": cache_reason,
                "restart_required": False,
                "core_probe": core_msg,
                "kernel_executable": sys.executable,
            }
            print(f"[setup] cache hit -> skipping deterministic setup: {core_msg}")

    if setup_result is None:
        if cache_reason and (not cache_hit):
            print(f"[setup] cache bypassed: {cache_reason}")
        setup_result = colab_setup.run_deterministic_setup(
            force_reinstall=force_reinstall,
            auto_restart=auto_restart_after_setup,
            strict_lock=strict_lockfile_check,
        )

    if (not bool(setup_result.get("restart_required", False))) and setup_cache_enabled:
        payload = {
            "status": "ready",
            "fingerprint": fingerprint,
            "timestamp_utc": int(time.time()),
            "kernel_executable": sys.executable,
            "repo_rev": str(repo_rev),
            "strict_lock": bool(strict_lockfile_check),
        }
        cache_path.write_text(json.dumps(payload, indent=2))
        print(f"[setup] cached ready state at {cache_path}")

    return SetupResult(
        result=setup_result,
        cache_hit=bool(cache_hit),
        cache_reason=str(cache_reason),
        cache_path=str(cache_path),
    )


def prepare_repo_imports(repo_dir: str = "/content/waymax-simulation-experiments", force_module_hot_reload: bool = True) -> str:
    repo_path = Path(repo_dir)
    if not repo_path.exists():
        raise RuntimeError(f"Repo checkout missing at {repo_dir}. Run Step 1 first.")

    os.chdir(str(repo_path))
    for path in (str(repo_path), str(repo_path / "src")):
        if path not in sys.path:
            sys.path.insert(0, path)

    if bool(force_module_hot_reload):
        for mod in [m for m in list(sys.modules) if (m == "src" or m.startswith("src."))]:
            sys.modules.pop(mod, None)
        importlib.invalidate_caches()

    return str(repo_path)


def bootstrap_colab_runtime(
    repo_url: str,
    repo_dir: str = "/content/waymax-simulation-experiments",
    repo_branch: str = "main",
    required_drive_folder: str = "/content/drive/MyDrive/waymax_experiments",
    verify_drive_access_every_run: bool = False,
    force_reinstall: bool = False,
    auto_restart_after_setup: bool = True,
    strict_lockfile_check: bool = True,
    setup_cache_enabled: bool = True,
    revalidate_core_imports_on_cache_hit: bool = True,
    setup_cache_path: str = "/content/.closedloop_setup_cache.json",
    force_module_hot_reload: bool = True,
) -> RuntimeBootstrapResult:
    repo_sync = ensure_repo_checkout(repo_url=repo_url, repo_dir=repo_dir, branch=repo_branch)
    drive_status = ensure_drive_ready(
        required_drive_folder=required_drive_folder,
        verify_drive_access_every_run=bool(verify_drive_access_every_run),
    )
    setup = run_cached_deterministic_setup(
        repo_root=repo_sync.repo_dir,
        force_reinstall=bool(force_reinstall),
        auto_restart_after_setup=bool(auto_restart_after_setup),
        strict_lockfile_check=bool(strict_lockfile_check),
        setup_cache_enabled=bool(setup_cache_enabled),
        revalidate_core_imports_on_cache_hit=bool(revalidate_core_imports_on_cache_hit),
        setup_cache_path=setup_cache_path,
        repo_rev=repo_sync.repo_rev,
    )
    prepared_repo_dir = prepare_repo_imports(
        repo_dir=repo_sync.repo_dir,
        force_module_hot_reload=bool(force_module_hot_reload),
    )
    return RuntimeBootstrapResult(
        repo_sync=repo_sync,
        drive_status=drive_status,
        setup=setup,
        prepared_repo_dir=prepared_repo_dir,
    )


def bootstrap_colab_runtime_with_config(config: ColabRuntimeConfig) -> RuntimeBootstrapResult:
    return bootstrap_colab_runtime(**config.to_bootstrap_kwargs())
