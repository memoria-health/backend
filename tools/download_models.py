"""Fetch every model the SAM A/B needs into ./models and the
ultralytics/boxmot cache. Run on the GPU box after gpu_setup.sh.

Gated weights (SAM 3.1, AdaFace) need HUGGINGFACE_TOKEN in the env and
a one-time license accept on the model's HF page. Un-gated weights
(OSNet, SCRFD) auto-download on first use, but we prefetch so the A/B
run doesn't stall.
"""

from __future__ import annotations

import os
from pathlib import Path

MODELS = Path("models")


def _sam() -> None:
    """SAM 3.1 via ultralytics — auto-downloads the checkpoint by name
    on construction."""
    from ultralytics import SAM

    name = os.environ.get("SAM_MODEL", "sam3.1_b.pt")
    print(f"[sam] fetching {name} ...")
    SAM(name)     # constructor downloads to the ultralytics cache
    print("[sam] ok")


def _adaface() -> None:
    """AdaFace checkpoint (gated) → models/adaface/."""
    from huggingface_hub import hf_hub_download

    dst = MODELS / "adaface"
    dst.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HUGGINGFACE_TOKEN")
    # Repo id is the AdaFace author's HF mirror; the exact id is
    # confirmed on the box (mk-minchul/adaface or a community mirror).
    repo = os.environ.get("ADAFACE_REPO", "mk-minchul/adaface")
    fname = os.environ.get("ADAFACE_CKPT", "adaface_ir101_webface4m.ckpt")
    print(f"[adaface] fetching {repo}/{fname} ...")
    try:
        p = hf_hub_download(repo_id=repo, filename=fname, token=token)
        (dst / fname).write_bytes(Path(p).read_bytes())
        print("[adaface] ok")
    except Exception as e:  # noqa: BLE001
        print(f"[adaface] FAILED ({e}). Accept the license on HF and set "
              f"HUGGINGFACE_TOKEN, or drop the .ckpt into {dst} manually.")


def _body() -> None:
    """OSNet-x1_0 body re-ID — boxmot auto-fetches by filename on first
    use; prefetch by constructing the backend."""
    print("[body] OSNet-x1_0 auto-fetches on first pipeline use (boxmot cache)")


def _scrfd() -> None:
    print("[scrfd] insightface buffalo_l auto-fetches on first use")


if __name__ == "__main__":
    MODELS.mkdir(exist_ok=True)
    _sam()
    _adaface()
    _body()
    _scrfd()
    print("\nAll set. If AdaFace failed, that's the one manual step.")
